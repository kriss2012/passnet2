import numpy as np
import pickle
import os
from flask import Flask, request, jsonify, render_template
# IMPORTANT: Using standard keras import for compatibility
from keras.models import load_model 
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # Hide messy logs

# --- MEMORY FIX ---
import tensorflow as tf
try:
    # Tell TensorFlow to barely use any memory at startup
    tf.config.set_visible_devices([], 'GPU') 
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        tf.config.experimental.set_memory_growth(device, True)
except:
    pass
# ------------------

app = Flask(__name__)

# --- Load Model Assets Safely ---
model = None
tokenizer = None

print("Server Starting... Loading assets.")
try:
    # Check if files exist before loading
    if os.path.exists('passnet_model.h5') and os.path.exists('tokenizer.pickle'):
        model = load_model('passnet_model.h5')
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        print("SUCCESS: Model and Tokenizer loaded.")
    else:
        print("ERROR: passnet_model.h5 or tokenizer.pickle not found.")
except Exception as e:
    print(f"CRITICAL ERROR loading model: {e}")

MAX_LEN = 20

def predict_strength(password):
    if not model or not tokenizer:
        return 0.0 # Fail safe if model isn't loaded
        
    seq = tokenizer.texts_to_sequences([password])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    prediction = model.predict(padded)[0][0]
    return float(prediction)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check():
    if not model:
        return jsonify({
            'score': 0, 
            'feedback': "Server Error: AI Model not loaded. Please check server logs.", 
            'color': 'red'
        }), 500

    data = request.json
    password = data.get('password', '')
    
    if not password:
        return jsonify({'error': 'No password provided'})

    try:
        score = predict_strength(password)
        
        # Interpret Score
        if score < 0.5:
            feedback = "Weak! Common pattern detected."
            color = "#ff4d4d" # Red
        elif score < 0.8:
            feedback = "Moderate. Add special characters or numbers."
            color = "#ffa600" # Orange
        else:
            feedback = "Strong! High entropy detected."
            color = "#2ecc71" # Green

        return jsonify({
            'score': round(score * 100, 1),
            'feedback': feedback,
            'color': color
        })
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    # Threaded=True helps prevent blocking
    app.run(debug=True, threaded=True)