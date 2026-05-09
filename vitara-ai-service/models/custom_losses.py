import tensorflow as tf

class WeightedFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, class_weights=None, name='weighted_focal_loss'):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha
        self.class_weights = class_weights # Dictionary {class_id: weight}

    def call(self, y_true, y_pred):
        # Ensure y_true is cast to float32 for calculations
        y_true = tf.cast(y_true, dtype=tf.int32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        # Convert sparse labels to one-hot encoding if needed
        if y_true.shape.ndims == y_pred.shape.ndims - 1:
            y_true = tf.one_hot(y_true, depth=y_pred.shape[-1])

        # Clip predictions to avoid log(0) errors
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        # Calculate cross-entropy part
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Calculate alpha_t and pt
        p_t = y_true * y_pred + (1 - y_true) * (1 - self.alpha)
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)

        # Apply class weights if provided
        if self.class_weights is not None:
            # Get true class indices from y_true (one-hot)
            true_class_indices = tf.argmax(y_true, axis=-1)
            # Map class indices to weights
            weights_tensor = tf.gather(tf.constant(list(self.class_weights.values()), dtype=tf.float32), true_class_indices)
            # Reshape weights to match cross_entropy shape for broadcasting
            weights_tensor = tf.expand_dims(weights_tensor, axis=-1)
            alpha_factor = alpha_factor * weights_tensor

        # Calculate focal loss
        focal_loss = alpha_factor * tf.math.pow((1. - p_t), self.gamma) * cross_entropy

        return tf.reduce_sum(focal_loss, axis=-1)