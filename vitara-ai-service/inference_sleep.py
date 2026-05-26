import json
import numpy as np
import tensorflow as tf
import joblib

def run_inference(duration_hours, interruptions, sleep_debt_hours):
    # Reload model dan scaler yang disimpan
    loaded_model = tf.keras.models.load_model('sleep_scoring_model.h5', compile=False)
    loaded_scaler = joblib.load('sleep_scaler.pkl')

    # Bungkus data mentah ke array 2 dimensi (3 fitur)
    raw_input = np.array([[duration_hours, interruptions, sleep_debt_hours]])

    # Lakukan transformasi scaling fitur
    scaled_input = loaded_scaler.transform(raw_input)

    # Prediksi nilai kualitas tidur
    prediction = loaded_model.predict(scaled_input, verbose=0)

    # Ambil nilai skalar dari numpy array hasil prediksi
    score_value = float(prediction[0][0])

    # Batasi skor pada rentang logis data [0.0 s.d 1.0]
    score_value = max(0.0, min(1.0, score_value))

    # Konversi hasil akhir menjadi format JSON string
    output_dict = {"quality_score": [round(score_value, 2)]}
    return json.dumps(output_dict)
