import os
import sys
import argparse
import numpy as np

# Dynamically add the vitara-ai-service directory to sys.path so we can import the custom layers/losses
current_dir = os.path.dirname(os.path.abspath(__file__))
service_dir = os.path.abspath(os.path.join(current_dir, "..", "vitara-ai-service"))
if service_dir not in sys.path:
    sys.path.append(service_dir)

def main():
    parser = argparse.ArgumentParser(description="Convert Keras model to ONNX format and verify correctness.")
    parser.add_argument(
        "--input", 
        type=str, 
        default=os.path.join(service_dir, "models", "nlp_model", "vitara_nlp_indobert_20260522_0846.keras"),
        help="Path to the input Keras (.keras or .h5) model file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=os.path.join(service_dir, "models", "nlp_model", "vitara_nlp_indobert.onnx"),
        help="Path where the output ONNX model file will be saved"
    )
    parser.add_argument(
        "--opset", 
        type=int, 
        default=15, 
        help="ONNX opset version (default: 15 for transformer compatibility)"
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    print("=" * 60)
    print(" Vitara AI Model Converter: Keras to ONNX ")
    print("=" * 60)
    print(f"Input Keras Model : {input_path}")
    print(f"Output ONNX Model : {output_path}")
    print(f"ONNX Opset Version: {args.opset}")
    print("-" * 60)

    if not os.path.exists(input_path):
        print(f"Error: Keras model file not found at {input_path}")
        sys.exit(1)

    # Import TensorFlow inside main so that running --help doesn't require importing TF (which is slow)
    print("Loading TensorFlow & Keras...")
    try:
        import tensorflow as tf
        import keras
    except ImportError:
        print("Error: TensorFlow or Keras is not installed in the current environment.")
        print("Please install them using: pip install tensorflow")
        sys.exit(1)

    # In TensorFlow 2.16+, tf.keras points to Keras 3.
    # The model was saved with Keras 3 configuration options (like 'optional': False in InputLayer)
    # but uses 'tf_keras.src.engine.functional.Functional' namespace in some environments.
    # To fix this, we load using Keras 3 (default) and map the legacy "Functional" class name
    # to Keras 3's Functional/Model class.
    try:
        from keras.src.models.functional import Functional
    except ImportError:
        try:
            from keras.src.engine.functional import Functional
        except ImportError:
            Functional = keras.Model

    # Helper to build a map of output shapes and dtypes for all layers in a Functional model config
    def infer_layer_outputs(layers):
        shapes = {}
        dtypes = {}
        for layer in layers:
            name = layer.get("name")
            class_name = layer.get("class_name")
            config = layer.get("config", {})
            
            # Default fallbacks
            shape = [None, 128]
            dtype = "float32"
            
            if class_name == "InputLayer":
                shape = config.get("batch_input_shape", [None, 128])
                dtype = config.get("dtype", "int32")
            elif class_name == "IndoBERTEncoder":
                shape = [None, 768]
                dtype = "float32"
            elif class_name == "Dense":
                units = config.get("units", 64)
                shape = [None, units]
                dtype = config.get("dtype", "float32")
                
            # Inherit from parent layer connection if present
            inbound = layer.get("inbound_nodes", [])
            if inbound and isinstance(inbound, list) and isinstance(inbound[0], list):
                first_node = inbound[0]
                if first_node and isinstance(first_node[0], list) and len(first_node[0]) > 0:
                    parent_name = first_node[0][0]
                    if parent_name in shapes:
                        # For layers that do not change dimension shape/dtype (e.g. Dropout, BatchNormalization)
                        if class_name not in ["Dense", "IndoBERTEncoder"]:
                            shape = shapes[parent_name]
                        dtype = dtypes[parent_name]
                        
            shapes[name] = shape
            dtypes[name] = dtype
        return shapes, dtypes

    # Helper to convert Keras 2 legacy inbound_nodes format to Keras 3 format
    def convert_legacy_inbound_nodes(inbound_nodes, shapes_map, dtypes_map):
        if not isinstance(inbound_nodes, list):
            return inbound_nodes
            
        print(f"DEBUG convert_legacy_inbound_nodes input: {inbound_nodes}")
        new_nodes = []
        for node in inbound_nodes:
            # Already in Keras 3 format
            if isinstance(node, dict) and ("args" in node or "kwargs" in node):
                new_nodes.append(node)
                continue
                
            # Dictionary input (typical for BERT layers taking input dicts)
            if isinstance(node, dict):
                converted_inputs = {}
                for k, v in node.items():
                    if isinstance(v, list) and len(v) >= 3:
                        parent_name = v[0]
                        converted_inputs[k] = {
                            "class_name": "__keras_tensor__",
                            "config": {
                                "shape": shapes_map.get(parent_name, [None, 128]),
                                "dtype": dtypes_map.get(parent_name, "float32"),
                                "keras_history": [v[0], v[1], v[2]]
                            }
                        }
                    else:
                        converted_inputs[k] = v
                new_nodes.append({
                    "args": [],
                    "kwargs": {
                        "inputs": converted_inputs
                    }
                })
            # List node (positional argument connections)
            elif isinstance(node, list):
                converted_args = []
                for arg in node:
                    if isinstance(arg, list) and len(arg) >= 3:
                        parent_name = arg[0]
                        converted_args.append({
                            "class_name": "__keras_tensor__",
                            "config": {
                                "shape": shapes_map.get(parent_name, [None, 128]),
                                "dtype": dtypes_map.get(parent_name, "float32"),
                                "keras_history": [arg[0], arg[1], arg[2]]
                            }
                        })
                    else:
                        converted_args.append(arg)
                new_nodes.append({
                    "args": converted_args,
                    "kwargs": {}
                })
            else:
                new_nodes.append(node)
        print(f"DEBUG convert_legacy_inbound_nodes output: {new_nodes}")
        return new_nodes

    # Low-level patch for Keras 3 deserialization to handle Keras 2 incompatibility issues
    deserialize_functions = []
    
    # 1. keras.saving.deserialize_keras_object
    if hasattr(keras, "saving") and hasattr(keras.saving, "deserialize_keras_object"):
        deserialize_functions.append((keras.saving, "deserialize_keras_object"))
        
    # 2. keras.src.saving.serialization_lib
    try:
        import keras.src.saving.serialization_lib as serialization_lib
        if hasattr(serialization_lib, "deserialize_keras_object"):
            deserialize_functions.append((serialization_lib, "deserialize_keras_object"))
    except ImportError:
        pass
        
    # 3. keras.src.saving.saving_lib (this is where _model_from_config imports and calls it)
    try:
        import keras.src.saving.saving_lib as saving_lib
        if hasattr(saving_lib, "deserialize_keras_object"):
            deserialize_functions.append((saving_lib, "deserialize_keras_object"))
    except ImportError:
        pass

    # 4. keras.src.saving.saving_api
    try:
        import keras.src.saving.saving_api as saving_api
        if hasattr(saving_api, "deserialize_keras_object"):
            deserialize_functions.append((saving_api, "deserialize_keras_object"))
    except ImportError:
        pass

    # 5. keras.saving.serialization_lib (fallback check)
    try:
        import keras.saving.serialization_lib as serialization_lib2
        if hasattr(serialization_lib2, "deserialize_keras_object"):
            deserialize_functions.append((serialization_lib2, "deserialize_keras_object"))
    except ImportError:
        pass

    # Apply the patch to all located deserializers
    for module_obj, func_name in deserialize_functions:
        orig_func = getattr(module_obj, func_name)
        
        def make_patched(original_func):
            def patched(identifier, *args, **kwargs):
                if isinstance(identifier, dict):
                    class_name = identifier.get("class_name")
                    print(f"DEBUG patched class_name: {class_name}")
                    config = identifier.get("config")
                    
                    # 1. Patch BatchNormalization axis list -> integer/tuple
                    if class_name == "BatchNormalization" and isinstance(config, dict):
                        axis = config.get("axis")
                        if isinstance(axis, list):
                            if len(axis) == 1:
                                config["axis"] = axis[0]
                            else:
                                config["axis"] = tuple(axis)
                                
                    # 2. Patch Functional Model layer inbound_nodes from Keras 2 to Keras 3 format
                    if class_name in ["Functional", "Model"] and isinstance(config, dict):
                        layers = config.get("layers", [])
                        shapes_map, dtypes_map = infer_layer_outputs(layers)
                        print(f"DEBUG Functional layers count: {len(layers)}")
                        for layer in layers:
                            if "inbound_nodes" in layer:
                                layer["inbound_nodes"] = convert_legacy_inbound_nodes(
                                    layer["inbound_nodes"], shapes_map, dtypes_map
                                )
                                
                    # 3. Pop quantization_config if present
                    if isinstance(config, dict) and 'quantization_config' in config:
                        config.pop('quantization_config', None)
                return original_func(identifier, *args, **kwargs)
            return patched
            
        setattr(module_obj, func_name, make_patched(orig_func))
    print("Keras deserializer patched successfully for BatchNormalization and legacy Functional model compatibility.")

    print("Loading Custom Layers & Losses...")
    
    # 1. AttentionLayer & WeightedFocalLoss from the project
    try:
        # pyrefly: ignore [missing-import]
        from models.custom_layers import AttentionLayer
        # pyrefly: ignore [missing-import]
        from models.custom_losses import WeightedFocalLoss
    except ImportError as e:
        print(f"Warning: Could not import project custom layers/losses: {e}")
        AttentionLayer = None
        WeightedFocalLoss = None

    # 2. IndoBERT Model Custom Layers
    def approximate_gelu(x):
        return tf.keras.activations.gelu(x, approximate=True)

    @keras.saving.register_keras_serializable(name="IndoBERTEncoder")
    class IndoBERTEncoder(keras.layers.Layer):
        def __init__(self, model_name='indobenchmark/indobert-base-p2', **kwargs):
            super().__init__(**kwargs)
            self.model_name = model_name
            # Import transformers inside __init__ to avoid startup overhead
            from transformers import TFBertModel, BertConfig
            config = BertConfig.from_pretrained(model_name)
            config.hidden_act = "gelu_new"  # Force 'gelu_new' (tanh approximation)
            self.bert = TFBertModel.from_pretrained(model_name, config=config)

        def call(self, inputs, training=False):
            if isinstance(inputs, dict):
                ids = inputs.get('input_ids')
                mask = inputs.get('attention_mask')
                tt = inputs.get('token_type_ids')
            elif isinstance(inputs, (list, tuple)):
                ids = inputs[0]
                mask = inputs[1] if len(inputs) > 1 else None
                tt = inputs[2] if len(inputs) > 2 else None
            else:
                ids = inputs
                mask = None
                tt = None
            
            # Fallbacks if mask or tt are None
            if mask is None:
                mask = tf.ones_like(ids)
            if tt is None:
                tt = tf.zeros_like(ids)
                
            out = self.bert(input_ids=ids, attention_mask=mask, token_type_ids=tt, training=training)
            return out.last_hidden_state[:, 0, :]

        def get_config(self):
            config = super().get_config()
            config.update({
                "model_name": self.model_name
            })
            return config

    # 3. Typing Model Custom Layers
    @keras.saving.register_keras_serializable(name="NotEqual")
    class NotEqual(keras.layers.Layer):
        def __init__(self, **kwargs):
            super(NotEqual, self).__init__(**kwargs)
        def __call__(self, *args, **kwargs):
            tensor_input = args[0]
            return super(NotEqual, self).__call__(tensor_input, **kwargs)
        def call(self, x):
            return tf.math.not_equal(x, 0.0)

    @keras.saving.register_keras_serializable(name="Any")
    class Any(keras.layers.Layer):
        def __init__(self, axis=-1, keepdims=False, **kwargs):
            super(Any, self).__init__(**kwargs)
            self.axis = axis
            self.keepdims = keepdims
        def call(self, x):
            return tf.reduce_any(x, axis=self.axis, keepdims=self.keepdims)
        def get_config(self):
            config = super(Any, self).get_config()
            config.update({'axis': self.axis, 'keepdims': self.keepdims})
            return config

    @keras.saving.register_keras_serializable(package="keras.layers", name="Dense")
    class PatchedDense(keras.layers.Dense):
        def __init__(self, *args, **kwargs):
            kwargs.pop('quantization_config', None)
            super(PatchedDense, self).__init__(*args, **kwargs)

    @keras.saving.register_keras_serializable(package="keras.layers", name="LSTM")
    class PatchedLSTM(keras.layers.LSTM):
        def __init__(self, *args, **kwargs):
            kwargs.pop('quantization_config', None)
            super(PatchedLSTM, self).__init__(*args, **kwargs)

    @keras.saving.register_keras_serializable(package="keras.layers", name="Masking")
    class PatchedMasking(keras.layers.Masking):
        def __init__(self, *args, **kwargs):
            kwargs.pop('quantization_config', None)
            super(PatchedMasking, self).__init__(*args, **kwargs)

    @keras.saving.register_keras_serializable(package="keras.layers", name="BatchNormalization")
    class PatchedBatchNormalization(keras.layers.BatchNormalization):
        def __init__(self, *args, **kwargs):
            if 'axis' in kwargs and isinstance(kwargs['axis'], list):
                if len(kwargs['axis']) == 1:
                    kwargs['axis'] = kwargs['axis'][0]
                else:
                    kwargs['axis'] = tuple(kwargs['axis'])
            super().__init__(*args, **kwargs)

    # Monkey patch globally
    keras.layers.Dense = PatchedDense
    tf.keras.layers.Dense = PatchedDense
    
    keras.layers.LSTM = PatchedLSTM
    tf.keras.layers.LSTM = PatchedLSTM
    
    keras.layers.Masking = PatchedMasking
    tf.keras.layers.Masking = PatchedMasking

    keras.layers.BatchNormalization = PatchedBatchNormalization
    tf.keras.layers.BatchNormalization = PatchedBatchNormalization

    # Register all possible custom objects, including the "Functional" model wrapper mapping
    custom_objects = {
        "Functional": Functional,
        "AttentionLayer": AttentionLayer,
        "WeightedFocalLoss": WeightedFocalLoss,
        "IndoBERTEncoder": IndoBERTEncoder,
        "gelu": approximate_gelu,
        "GELU": approximate_gelu,
        "NotEqual": NotEqual,
        "Any": Any,
        "Dense": PatchedDense,
        "LSTM": PatchedLSTM,
        "Masking": PatchedMasking,
        "BatchNormalization": PatchedBatchNormalization
    }

    print("Loading Keras model...")
    try:
        # Load with compile=False to ignore compile-time configurations like losses & optimizers
        model = keras.models.load_model(input_path, custom_objects=custom_objects, compile=False)
        print("Keras model loaded successfully.")
    except Exception as e:
        import traceback
        print(f"Error loading Keras model: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("Converting model to ONNX...")
    try:
        # pyrefly: ignore [missing-import]
        import tf2onnx
    except ImportError:
        print("Error: tf2onnx is not installed.")
        print("Please install it using: pip install tf2onnx")
        sys.exit(1)

    # Dynamically build the input signature from model.inputs
    input_signature = []
    resolved_inputs = []
    print("Detected model inputs:")
    for inp in model.inputs:
        if isinstance(inp, str):
            # Input is represented as a string name in some Keras 3 configurations
            input_layer = None
            for layer in model.layers:
                if layer.name == inp:
                    input_layer = layer
                    break
            
            if input_layer is not None:
                if hasattr(input_layer, 'batch_shape'):
                    shape = input_layer.batch_shape
                elif hasattr(input_layer, 'batch_input_shape'):
                    shape = input_layer.batch_input_shape
                else:
                    shape = input_layer.input_shape
                
                dtype = getattr(input_layer, 'dtype', 'int32')
                tf_dtype = tf.as_dtype(dtype)
                dtype_name = tf_dtype.name
                print(f"  Name: '{inp}' | Shape: {shape} | Dtype: {dtype_name}")
                spec = tf.TensorSpec(shape=shape, dtype=tf_dtype, name=inp)
            else:
                print(f"  Name: '{inp}' (could not resolve layer config, using defaults)")
                spec = tf.TensorSpec(shape=(None, 128), dtype=tf.int32, name=inp)
        else:
            name = inp.name.split(':')[0] if hasattr(inp, 'name') else str(inp)
            tf_dtype = tf.as_dtype(inp.dtype)
            dtype_name = tf_dtype.name
            print(f"  Name: '{name}' | Shape: {inp.shape} | Dtype: {dtype_name}")
            spec = tf.TensorSpec(shape=inp.shape, dtype=tf_dtype, name=name)
            
        input_signature.append(spec)
        resolved_inputs.append(spec)

    try:
        onnx_model, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=input_signature,
            opset=args.opset
        )
        print("ONNX model conversion complete.")
    except Exception as e:
        print(f"Error converting model: {e}")
        sys.exit(1)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print("Saving ONNX model...")
    try:
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"ONNX model saved successfully to: {output_path}")
    except Exception as e:
        print(f"Error saving ONNX model: {e}")
        sys.exit(1)

    # Verification Step using ONNX Runtime
    print("\n" + "=" * 60)
    print(" Verification Phase ")
    print("=" * 60)
    print("Loading ONNX Runtime for verification...")
    try:
        # pyrefly: ignore [missing-import]
        import onnxruntime as ort
    except ImportError:
        print("Warning: onnxruntime is not installed. Skipping verification step.")
        print("You can verify manually or install onnxruntime: pip install onnxruntime")
        print("=" * 60)
        sys.exit(0)

    try:
        # Generate dynamic matching dummy inputs
        dummy_inputs = {}
        keras_input_list = []
        
        for spec in resolved_inputs:
            name = spec.name
            shape = spec.shape
            
            # Resolve None dimension sizes to 1 for dummy run (e.g. batch size)
            resolved_shape = []
            for dim in shape:
                if dim is None:
                    resolved_shape.append(1)
                else:
                    resolved_shape.append(dim)
            
            # Generate random values depending on input type
            if spec.dtype.is_integer:
                if "token_type_ids" in name or "attention_mask" in name:
                    # Segment IDs and attention masks must be 0 or 1
                    dummy_val = np.random.randint(0, 2, size=resolved_shape, dtype=spec.dtype.as_numpy_dtype)
                else:
                    # Token IDs (within vocab range)
                    dummy_val = np.random.randint(0, 10000, size=resolved_shape, dtype=spec.dtype.as_numpy_dtype)
            else:
                # Float inputs
                dummy_val = np.random.randn(*resolved_shape).astype(spec.dtype.as_numpy_dtype)
                
            dummy_inputs[name] = dummy_val
            keras_input_list.append(dummy_val)

        # Run prediction on Keras model
        print("Running dummy inference with Keras model...")
        if len(keras_input_list) == 1:
            keras_pred = model.predict(keras_input_list[0])
        else:
            keras_pred = model.predict(keras_input_list)

        # Run prediction on ONNX model
        print("Running dummy inference with ONNX model...")
        ort_sess = ort.InferenceSession(output_path)
        
        # Match ONNX inputs to our generated dummy values
        ort_inputs = {}
        for ort_in in ort_sess.get_inputs():
            name = ort_in.name
            if name in dummy_inputs:
                ort_inputs[name] = dummy_inputs[name]
            else:
                # Fuzzy match if names are slightly modified
                matched = False
                for k in dummy_inputs.keys():
                    if k in name or name in k:
                        ort_inputs[name] = dummy_inputs[k]
                        matched = True
                        break
                if not matched:
                    # Fallback by indexing
                    pass
        
        # Fallback if names didn't match
        if len(ort_inputs) < len(ort_sess.get_inputs()):
            for i, ort_in in enumerate(ort_sess.get_inputs()):
                if ort_in.name not in ort_inputs:
                    key = list(dummy_inputs.keys())[i]
                    ort_inputs[ort_in.name] = dummy_inputs[key]

        onnx_preds = ort_sess.run(None, ort_inputs)

        # Compare outputs
        if isinstance(keras_pred, list) or isinstance(keras_pred, tuple):
            # Multiple outputs
            print("\nKeras Outputs:")
            for i, kp in enumerate(keras_pred):
                print(f"  Output {i} Shape: {kp.shape}")
            print("ONNX Outputs:")
            for i, op in enumerate(onnx_preds):
                print(f"  Output {i} Shape: {op.shape}")
                
            all_match = True
            for i, (kp, op) in enumerate(zip(keras_pred, onnx_preds)):
                diff = np.max(np.abs(kp - op))
                print(f"Output {i} - Maximum absolute difference: {diff:.6e}")
                if diff >= 1e-4:
                    all_match = False
            
            if all_match:
                print("\n✅ SUCCESS: ONNX model outputs match Keras model outputs!")
            else:
                print("\n⚠️ WARNING: Output mismatch exceeds threshold (1e-4) on one or more outputs.")
        else:
            # Single output
            onnx_pred = onnx_preds[0]
            print("\nKeras Output Shape:", keras_pred.shape)
            print("ONNX Output Shape :", onnx_pred.shape)
            
            diff = np.max(np.abs(keras_pred - onnx_pred))
            print(f"Maximum absolute difference: {diff:.6e}")
            
            if diff < 1e-4:
                print("\n✅ SUCCESS: ONNX model output matches Keras model output!")
            else:
                print("\n⚠️ WARNING: Output mismatch exceeds threshold (1e-4).")

    except Exception as e:
        print(f"Error during verification: {e}")

    print("=" * 60)

if __name__ == "__main__":
    main()
