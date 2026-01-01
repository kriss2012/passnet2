import pandas as pd
import random
import string
import numpy as np

# CONFIG
MAX_PASSWORDS = 200000  # We'll use top 200k from rockyou to save training time. Increase if you have a GPU.
INPUT_FILE = 'rockyou.txt'
OUTPUT_FILE = 'training_data.csv'

def generate_strong_password():
    """Generates a random strong password."""
    length = random.randint(10, 16)
    chars = string.ascii_letters + string.digits + "!@#$%^&*"
    return ''.join(random.choice(chars) for _ in range(length))

def create_dataset():
    print(f"Loading {INPUT_FILE}... this might take a moment.")
    
    # 1. Load Weak Passwords (Class 0)
    # handling encoding errors common in rockyou.txt
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            # Read line by line to save memory, take top N
            weak_passwords = [next(f).strip() for _ in range(MAX_PASSWORDS)]
    except StopIteration:
        pass # File is smaller than MAX_PASSWORDS
        
    print(f"Loaded {len(weak_passwords)} weak passwords.")

    # 2. Generate Strong Passwords (Class 1)
    print("Generating synthetic strong passwords...")
    strong_passwords = [generate_strong_password() for _ in range(len(weak_passwords))]

    # 3. Combine and Label
    data_weak = pd.DataFrame({'password': weak_passwords, 'label': 0})
    data_strong = pd.DataFrame({'password': strong_passwords, 'label': 1})
    
    full_data = pd.concat([data_weak, data_strong], axis=0).sample(frac=1).reset_index(drop=True)
    
    # 4. Save
    full_data.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset saved to {OUTPUT_FILE} with {len(full_data)} samples.")

if __name__ == "__main__":
    create_dataset()