import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# CONFIG
MAX_LEN = 20  # Max characters to consider
VOCAB_SIZE = 100  # Number of unique characters to learn (ASCII standard)

def train():
    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv('training_data.csv')
    df['password'] = df['password'].astype(str) # Ensure strings

    # 2. Tokenization (Convert Chars to Numbers)
    tokenizer = Tokenizer(char_level=True, lower=False, num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(df['password'])
    
    sequences = tokenizer.texts_to_sequences(df['password'])
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')
    y = df['label'].values

    # 3. Build Model
    print("Building Model...")
    model = Sequential([
        Input(shape=(MAX_LEN,)),
        Embedding(input_dim=VOCAB_SIZE, output_dim=32),
        LSTM(64, dropout=0.2), # LSTM learns sequences
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid') # Output 0-1
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 4. Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print("Training... (This will take a few minutes)")
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

    # 5. Save Artifacts
    model.save('passnet_model.h5')
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Model and Tokenizer saved successfully!")

if __name__ == "__main__":
    train()