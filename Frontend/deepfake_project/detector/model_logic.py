# File: detector/model_logic.py

import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from django.conf import settings

# --- 1. Load Model ONCE ---
CURRENT_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(CURRENT_DIR, 'best_model.h5')

print(f"--- Checking for model at: {MODEL_PATH} ---")
print(f"--- Does file exist? {os.path.exists(MODEL_PATH)} ---")

print(f"--- Loading model from {MODEL_PATH} ---")
model = load_model(MODEL_PATH)
print("--- Model loaded successfully ---")


# --- 2. THE CORRECT FUNCTION ---
def run_model_on_file(filepath):
    """
    This function is called by views.py and returns
    the dictionary that results.html needs.
    """
    
    try:
        img = image.load_img(filepath, target_size=(224, 224))
        
        # --- THIS IS THE FIX ---
        # We MUST normalize the image by dividing by 255.0
        # just like the training script did.
        img_array = image.img_to_array(img) / 255.0
        # --- END OF FIX ---

        img_array = np.expand_dims(img_array, axis=0)

        # --- Predict ---
        pred = model.predict(img_array)[0][0] 

        # --- Get Feedback ---
        if pred > 0.9:
            feedback = "Definitely Real"
            css_class = "real"
        elif pred > 0.7:
            feedback = "Likely Real"
            css_class = "real"
        elif pred > 0.3:
            feedback = "Uncertain"
            css_class = "uncertain"
        else:
            feedback = "Likely Fake"
            css_class = "fake"

        # --- Return the correct dictionary ---
        return {
            "prediction_score": pred,
            "feedback": feedback,
            "css_class": css_class
        }
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return {
            "error": str(e)
        }