import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime

# =================================================================
# VITARA AI - MODEL VALIDATION SCRIPT
# =================================================================

# Add the parent directory to sys.path to allow imports from vitara-ai-service
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
sys.path.append(BASE_DIR)

# Import custom components if they exist
CUSTOM_OBJECTS = {}

# Custom compatibility layers for robust Keras 3 deserialization
class NotEqual(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NotEqual, self).__init__(**kwargs)
    def __call__(self, *args, **kwargs):
        tensor_input = args[0]
        return super(NotEqual, self).__call__(tensor_input, **kwargs)
    def call(self, x):
        return tf.math.not_equal(x, 0.0)

class Any(tf.keras.layers.Layer):
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

class PatchedDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super(PatchedDense, self).__init__(*args, **kwargs)

class PatchedLSTM(tf.keras.layers.LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super(PatchedLSTM, self).__init__(*args, **kwargs)

class PatchedMasking(tf.keras.layers.Masking):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super(PatchedMasking, self).__init__(*args, **kwargs)

CUSTOM_OBJECTS.update({
    'NotEqual': NotEqual,
    'Any': Any,
    'Dense': PatchedDense,
    'LSTM': PatchedLSTM,
    'Masking': PatchedMasking
})

try:
    from models.custom_layers import AttentionLayer
    CUSTOM_OBJECTS['AttentionLayer'] = AttentionLayer
except ImportError:
    print("Warning: AttentionLayer not found.")

try:
    from models.custom_losses import WeightedFocalLoss
    CUSTOM_OBJECTS['WeightedFocalLoss'] = WeightedFocalLoss
except ImportError:
    print("Warning: WeightedFocalLoss not found.")

# ANSI Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}")
    print(f"{text.center(60)}")
    print(f"{'='*60}{Colors.ENDC}")

def print_result(name, value, target, condition='ge'):
    """
    Prints a formatted result and returns whether it passed.
    condition: 'ge' (>=) or 'le' (<=)
    """
    if condition == 'ge':
        passed = value >= target
        op = '>='
    else:
        passed = value <= target
        op = '<='
    
    color = Colors.OKGREEN if passed else Colors.FAIL
    status = "PASS" if passed else "FAIL"
    
    print(f"{Colors.BOLD}{name:35}{Colors.ENDC}: "
          f"{color}{value:.4f}{Colors.ENDC} (Target {op} {target:.4f}) -> {color}{status}{Colors.ENDC}")
    return passed

class ModelValidator:
    def __init__(self):
        self.results = {}
        print(f"{Colors.OKCYAN}Initializing Validator at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
        print(f"Base Directory: {BASE_DIR}")

    def validate_nlp(self):
        print_header("NLP Stress/Emotion Model Validation")
        model_path = os.path.join(BASE_DIR, "models/nlp_model")
        test_data_path = os.path.join(DATA_DIR, "nlp/processed/test.csv")
        
        if not os.path.exists(model_path):
            print(f"{Colors.WARNING}NLP Model not found at {model_path}{Colors.ENDC}")
            return
        
        if not os.path.exists(test_data_path):
            print(f"{Colors.WARNING}NLP Test data not found at {test_data_path}{Colors.ENDC}")
            return

        try:
            print(f"Loading model: {model_path}")
            model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)
            
            print(f"Loading test data: {test_data_path}")
            test_df = pd.read_csv(test_data_path)
            
            # Assuming 'text' is the input and ['emotion_label', 'stress_score'] are targets
            # This depends on the specific training pipeline
            x_test = test_df['text'].values
            y_emotion = test_df['emotion_label'].values
            y_stress = test_df['stress_score'].values
            
            print("Running model.evaluate()...")
            # The metrics depend on how the model was compiled
            results = model.evaluate(x_test, [y_emotion, y_stress], verbose=0)
            
            # Assuming metrics: [loss, emotion_acc, stress_acc]
            # We use the emotion accuracy as the primary metric for the threshold
            emotion_acc = results[1] if len(results) > 1 else results[0]
            
            p1 = print_result("Emotion/Stress Accuracy", emotion_acc, 0.85)
            self.results["NLP"] = p1
            
        except Exception as e:
            print(f"{Colors.FAIL}Error validating NLP: {e}{Colors.ENDC}")
            self.results["NLP"] = False

    def validate_vision(self):
        print_header("Food Vision Model Validation")
        model_dir = os.path.join(BASE_DIR, "models/vision_model")
        test_dir = os.path.join(DATA_DIR, "vision/processed/test")
        calorie_map_path = os.path.join(DATA_DIR, "vision/raw/calorie_map.csv")
        
        if not os.path.exists(model_dir):
            print(f"{Colors.WARNING}Vision Model directory not found at {model_dir}{Colors.ENDC}")
            return
        
        # Find .tflite file in model directory
        tflite_files = [f for f in os.listdir(model_dir) if f.endswith('.tflite')]
        if not tflite_files:
            print(f"{Colors.WARNING}No .tflite model found in {model_dir}{Colors.ENDC}")
            return
        tflite_path = os.path.join(model_dir, tflite_files[0])
        
        if not os.path.exists(test_dir):
            print(f"{Colors.WARNING}Vision Test directory not found at {test_dir}{Colors.ENDC}")
            return
            
        try:
            # pyrefly: ignore [missing-import]
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            # pyrefly: ignore [missing-import]
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            from PIL import Image

            print(f"Loading TFLite model: {tflite_path}")
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            print(f"Preparing test generator from: {test_dir}")
            test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
            test_generator = test_datagen.flow_from_directory(
                test_dir,
                target_size=(224, 224),
                batch_size=1,   # TFLite interpreter runs one sample at a time
                class_mode='sparse',
                shuffle=False
            )
            
            # Load calorie map if available
            idx_to_calorie = None
            if os.path.exists(calorie_map_path):
                print(f"Loading calorie map: {calorie_map_path}")
                calorie_df = pd.read_csv(calorie_map_path)
                class_to_calorie = dict(zip(calorie_df['class_name'], calorie_df['calories_per_100g']))
                MAX_CALORIES = 1000.0
                idx_to_calorie = {v: class_to_calorie.get(k, 0) / MAX_CALORIES
                                  for k, v in test_generator.class_indices.items()}

            print(f"Running TFLite inference on {test_generator.samples} test images...")
            correct = 0
            calorie_abs_errors = []
            
            for i in range(len(test_generator)):
                x_batch, y_true_batch = test_generator[i]
                
                interpreter.set_tensor(input_details[0]['index'], x_batch.astype(np.float32))
                interpreter.invoke()
                
                # Determine class vs calorie output by shape
                out_0 = interpreter.get_tensor(output_details[0]['index'])
                out_1 = interpreter.get_tensor(output_details[1]['index']) if len(output_details) > 1 else None
                
                if out_1 is not None:
                    if output_details[0]['shape'][-1] == 1:
                        calorie_pred, class_pred = out_0, out_1
                    else:
                        class_pred, calorie_pred = out_0, out_1
                else:
                    class_pred = out_0
                    calorie_pred = None
                
                predicted_class = int(np.argmax(class_pred, axis=-1)[0])
                true_class = int(y_true_batch[0])
                if predicted_class == true_class:
                    correct += 1
                
                if calorie_pred is not None and idx_to_calorie is not None:
                    true_cal = idx_to_calorie.get(true_class, 0)
                    pred_cal = float(calorie_pred[0][0])
                    calorie_abs_errors.append(abs(pred_cal - true_cal))
            
            acc = correct / test_generator.samples
            p1 = print_result("Classification Accuracy", acc, 0.85)
            
            if calorie_abs_errors:
                mae = float(np.mean(calorie_abs_errors))
                p2 = print_result("Calorie MAE (normalized)", mae, 0.02, condition='le')
                self.results["Vision"] = p1 and p2
            else:
                self.results["Vision"] = p1
            
        except Exception as e:
            print(f"{Colors.FAIL}Error validating Vision: {e}{Colors.ENDC}")
            self.results["Vision"] = False

    def validate_typing(self):
        print_header("Typing Stress Model Validation")
        typing_path = os.path.join(BASE_DIR, "models/typing_model")
        model_file = os.path.join(typing_path, "typing_stress_lstm.h5")
        if not os.path.exists(model_file):
            model_file = typing_path
            
        typing_test_path = os.path.join(DATA_DIR, "typing/processed/test.csv")
        typing_test_dir = os.path.join(DATA_DIR, "typing/processed/test")
        
        if os.path.exists(model_file) and (os.path.exists(typing_test_path) or os.path.exists(typing_test_dir)):
            try:
                print(f"Loading model: {model_file}")
                model = tf.keras.models.load_model(model_file, custom_objects=CUSTOM_OBJECTS)
                
                if os.path.exists(typing_test_dir):
                    print(f"Loading test numpy matrices from: {typing_test_dir}")
                    X_seq = np.load(os.path.join(typing_test_dir, "X_seq.npy"))
                    X_static = np.load(os.path.join(typing_test_dir, "X_static.npy"))
                    y = np.load(os.path.join(typing_test_dir, "y.npy"))
                    results = model.evaluate([X_seq, X_static], y, verbose=0)
                else:
                    print(f"Loading test data: {typing_test_path}")
                    test_df = pd.read_csv(typing_test_path)
                    results = model.evaluate(test_df.drop('target', axis=1), test_df['target'], verbose=0)
                    
                auc = results[1] if len(results) > 1 else results[0]
                self.results["Typing"] = print_result("Typing Stress AUC-ROC", auc, 0.80)
            except Exception as e:
                print(f"{Colors.FAIL}Error validating Typing: {e}{Colors.ENDC}")
                self.results["Typing"] = False
        else:
            print(f"{Colors.WARNING}Typing model or data missing. Skipping.{Colors.ENDC}")

    def validate_sleep(self):
        print_header("Sleep Scoring Model Validation")
        sleep_path = os.path.join(BASE_DIR, "models/sleep_model")
        sleep_test_path = os.path.join(DATA_DIR, "sleep/processed/test.csv")
        
        if os.path.exists(sleep_path) and os.path.exists(sleep_test_path):
            try:
                print(f"Loading model: {sleep_path}")
                model = tf.keras.models.load_model(sleep_path, custom_objects=CUSTOM_OBJECTS)
                print(f"Loading test data: {sleep_test_path}")
                test_df = pd.read_csv(sleep_test_path)
                results = model.evaluate(test_df.drop('target', axis=1), test_df['target'], verbose=0)
                mae = results[1] if len(results) > 1 else results[0]
                self.results["Sleep"] = print_result("Sleep Scoring MAE", mae, 0.02, condition='le')
            except Exception as e:
                print(f"{Colors.FAIL}Error validating Sleep: {e}{Colors.ENDC}")
                self.results["Sleep"] = False
        else:
            print(f"{Colors.WARNING}Sleep model or data missing. Skipping.{Colors.ENDC}")

    def validate_health_score(self):
        print_header("Rule-Based Health Score Validation")
        try:
            from services.health_score_service import HealthScoreService
            from schemas.health_score import HealthScoreRequest, NLPResult, FoodResult, SleepResult, TypingResult
            
            # Create a mock request representing a healthy state
            healthy_req = HealthScoreRequest(
                user_id="test_user_healthy",
                nlp_result=NLPResult(emotion="happy", stress_level=0.2),
                food_result=FoodResult(estimated_calories=550),
                sleep_result=SleepResult(quality_score=85),
                typing_result=TypingResult(stress_score=0.15)
            )
            
            # Create a mock request representing a poor health/stressed state
            stressed_req = HealthScoreRequest(
                user_id="test_user_stressed",
                nlp_result=NLPResult(emotion="sad", stress_level=0.8),
                food_result=FoodResult(estimated_calories=120),
                sleep_result=SleepResult(quality_score=45),
                typing_result=TypingResult(stress_score=0.75)
            )
            
            # Run calculations
            healthy_res = HealthScoreService.calculate_health_score(healthy_req)
            stressed_res = HealthScoreService.calculate_health_score(stressed_req)
            
            # Assertions for validation
            # 1. Logic check: Healthy overall score must be greater than stressed overall score
            logic_ok = healthy_res.health_score > stressed_res.health_score
            p1 = print_result("Healthy > Stressed (Logic Check)", 1.0 if logic_ok else 0.0, 1.0)
            
            # 2. Stress score conversion: avg stress = 0.175. stress_score should be round((1 - 0.175) * 100) = 83.
            p2 = print_result("Stress Calculation Check", float(healthy_res.breakdown.stress), 83.0)
            
            # 3. Mood score: happy = 90
            p3 = print_result("Mood Classification Check", float(healthy_res.breakdown.mood), 90.0)
            
            # 4. Sleep score: quality_score = 85
            p4 = print_result("Sleep Quality Mapping Check", float(healthy_res.breakdown.sleep), 85.0)
            
            # 5. Nutrition score: 550 kcal = 90
            p5 = print_result("Nutrition Calorie Mapping Check", float(healthy_res.breakdown.nutrition), 90.0)
            
            # 6. Overall weighted score check
            # (83 * 0.3) + (85 * 0.3) + (90 * 0.2) + (90 * 0.2) = 24.9 + 25.5 + 18 + 18 = 86.4 -> round to 86.
            p6 = print_result("Overall Health Score Check", float(healthy_res.health_score), 86.0)
            
            self.results["Health Score"] = p1 and p2 and p3 and p4 and p5 and p6
        except Exception as e:
            print(f"{Colors.FAIL}Error validating Health Score: {e}{Colors.ENDC}")
            self.results["Health Score"] = False

    def summary(self):
        print_header("VALIDATION SUMMARY")
        all_passed = True
        for model, passed in self.results.items():
            status = f"{Colors.OKGREEN}PASSED{Colors.ENDC}" if passed else f"{Colors.FAIL}FAILED{Colors.ENDC}"
            print(f"{model:20}: {status}")
            if not passed:
                all_passed = False
        
        if all_passed:
            print(f"\n{Colors.OKGREEN}{Colors.BOLD}ALL MODELS PASSED THRESHOLDS!{Colors.ENDC}")
        else:
            print(f"\n{Colors.FAIL}{Colors.BOLD}SOME MODELS FAILED VALIDATION.{Colors.ENDC}")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Vitara AI Model Validation Script")
    parser.add_argument("--nlp", action="store_true", help="Validate NLP model")
    parser.add_argument("--vision", action="store_true", help="Validate Vision model")
    parser.add_argument("--typing", action="store_true", help="Validate Typing model")
    parser.add_argument("--sleep", action="store_true", help="Validate Sleep model")
    parser.add_argument("--health", action="store_true", help="Validate Health Score model")
    parser.add_argument("--all", action="store_true", help="Validate all models")
    
    args = parser.parse_args()
    
    validator = ModelValidator()
    
    if args.all or args.nlp:
        validator.validate_nlp()
    if args.all or args.vision:
        validator.validate_vision()
    if args.all or args.typing:
        validator.validate_typing()
    if args.all or args.sleep:
        validator.validate_sleep()
    if args.all or args.health:
        validator.validate_health_score()
        
    if not (args.all or args.nlp or args.vision or args.typing or args.sleep or args.health):
        print(f"{Colors.WARNING}No models specified for validation. Use --all or specific model flags.{Colors.ENDC}")
        parser.print_help()
        return

    validator.summary()

if __name__ == "__main__":
    main()
