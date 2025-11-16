import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model
model = load_model('./tmp_checkpoint/best_model.h5')

# Load and preprocess image
import sys
img_path = sys.argv[1] if len(sys.argv) > 1 else './split_dataset/test/real/abarnvbtwb-000-00.png'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
pred = model.predict(img_array)[0][0]

# Feedback
if pred > 0.9:
    feedback = "🟢 Definitely Real"
elif pred > 0.7:
    feedback = "🟡 Likely Real"
elif pred > 0.3:
    feedback = "🟠 Uncertain"
else:
    feedback = "🔴 Likely Fake"

print(f"Prediction Score: {pred:.4f}")
print(f"Feedback: {feedback}")