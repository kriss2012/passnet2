import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load Model & Tokenizer globally
print("Loading model assets...")
model = load_model('passnet_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

MAX_LEN = 20

def predict_strength(password):
    # Preprocess input exactly like training data
    seq = tokenizer.texts_to_sequences([password])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    prediction = model.predict(padded)[0][0]
    return float(prediction)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check():
    data = request.json
    password = data.get('password', '')
    
    if not password:
        return jsonify({'error': 'No password provided'})

    score = predict_strength(password)
    
    # Interpret Score
    if score < 0.5:
        feedback = "Weak! Found patterns similar to leaked passwords."
        color = "red"
    elif score < 0.8:
        feedback = "Moderate. Adding special chars might help."
        color = "orange"
    else:
        feedback = "Strong! Good entropy and randomness."
        color = "green"

    return jsonify({
        'score': round(score * 100, 1),
        'feedback': feedback,
        'color': color
    })

if __name__ == '__main__':
    app.run(debug=True)
