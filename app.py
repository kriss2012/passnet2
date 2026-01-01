import numpy as np
import pickle
import os
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gc

app = Flask(__name__)

# --- CONFIGURATION ---
MAX_LEN = 20
MODEL_PATH = 'passnet_model.h5'
TOKENIZER_PATH = 'tokenizer.pickle'

# Global variables (Initially Empty)
model = None
tokenizer = None

def get_model():
    """
    Lazy Loading: Only load the heavy model when we actually need it.
    This prevents the server from crashing at startup.
    """
    global model, tokenizer
    
    # If already loaded, just return it
    if model and tokenizer:
        return model, tokenizer

    print("⚡ Loading AI Model for the first time...")
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
            model = load_model(MODEL_PATH)
            with open(TOKENIZER_PATH, 'rb') as handle:
                tokenizer = pickle.load(handle)
            print("✅ Model loaded successfully!")
            return model, tokenizer
        else:
            print("❌ ERROR: Model files not found.")
            return None, None
    except Exception as e:
        print(f"❌ CRITICAL LOAD ERROR: {e}")
        return None, None

@app.route('/')
def home():
    # The home page loads instantly because we aren't loading the AI yet
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check():
    # 1. Load the model NOW (if not already loaded)
    ai_model, ai_tokenizer = get_model()
    
    if not ai_model or not ai_tokenizer:
        return jsonify({
            'score': 0, 
            'feedback': "Server Error: Model files missing or too large for RAM.", 
            'color': 'red'
        }), 500

    # 2. Process Input
    data = request.json
    password = data.get('password', '')
    
    if not password:
        return jsonify({'error': 'No password provided'})

    try:
        # 3. Predict
        seq = ai_tokenizer.texts_to_sequences([password])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
        prediction = ai_model.predict(padded, verbose=0)[0][0] # verbose=0 hides logs
        
        score = float(prediction)
        
        # 4. Cleanup Memory (Crucial for Free Tier)
        # We force Python to clean up unused memory after every check
        gc.collect()

        # Interpret Score
        if score < 0.5:
            feedback = "Weak! Common pattern detected."
            color = "#ff4d4d"
        elif score < 0.8:
            feedback = "Moderate. Add special characters."
            color = "#ffa600"
        else:
            feedback = "Strong! High entropy detected."
            color = "#2ecc71"

        return jsonify({
            'score': round(score * 100, 1),
            'feedback': feedback,
            'color': color
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)