import tensorflow as tf
import time
import datetime

class VitaraTrainingLogger(tf.keras.callbacks.Callback):
    """
    Custom callback for Vitara AI models training.
    Provides detailed, formatted logging for each epoch, including time taken,
    loss, and metrics (like accuracy, MAE).
    
    Usage:
        from vitara_ai_service.models.custom_callbacks import VitaraTrainingLogger
        
        logger = VitaraTrainingLogger(log_frequency=1)
        model.fit(..., callbacks=[logger])
    """
    
    def __init__(self, log_frequency=1):
        super(VitaraTrainingLogger, self).__init__()
        self.log_frequency = log_frequency
        self.train_start_time = None
        self.epoch_start_time = None
        
    def on_train_begin(self, logs=None):
        self.train_start_time = time.time()
        print("="*80)
        # Set timezone to WIB (UTC+7)
        wib_tz = datetime.timezone(datetime.timedelta(hours=7))
        start_time_str = datetime.datetime.now(wib_tz).strftime('%Y-%m-%d %H:%M:%S WIB')
        print(f"🚀 [Vitara AI] Training Started at {start_time_str}")
        print("="*80)
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.log_frequency == 0:
            epoch_time = time.time() - self.epoch_start_time
            
            log_msgs = [f"Epoch {epoch + 1:03d} | {epoch_time:.1f}s"]
            
            if logs:
                for key, value in logs.items():
                    log_msgs.append(f"{key}: {value:.4f}")
                    
            print(" | ".join(log_msgs))
            
    def on_train_end(self, logs=None):
        total_time = time.time() - self.train_start_time
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print("="*80)
        print(f"✅ [Vitara AI] Training Completed in {int(hours):02d}:{int(minutes):02d}:{seconds:.1f}")
        print("="*80)
