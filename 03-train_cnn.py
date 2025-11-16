import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB0

print('TensorFlow version:', tf.__version__)

# Paths
dataset_path = r'.\split_dataset'
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'val')
test_path = os.path.join(dataset_path, 'test')
checkpoint_filepath = r'.\tmp_checkpoint'
os.makedirs(checkpoint_filepath, exist_ok=True)

# Parameters
input_size = (224, 224)
batch_size = 32
num_epochs = 20

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

print(train_generator.class_indices)  # ✅ Add this here

val_generator = val_datagen.flow_from_directory(
    val_path,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=input_size,
    batch_size=1,
    class_mode=None,
    shuffle=False
)

# Model setup
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# Callbacks
custom_callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min'),
    ModelCheckpoint(filepath=os.path.join(checkpoint_filepath, 'best_model.h5'),
                    monitor='val_loss', save_best_only=True, verbose=1, mode='min')
]

# Train
history = model.fit(
    train_generator,
    epochs=num_epochs,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=custom_callbacks
)

# Plot Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# Plot Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Load best model
best_model = load_model(os.path.join(checkpoint_filepath, 'best_model.h5'))

# Predict
test_generator.reset()
preds = best_model.predict(test_generator, verbose=1)

# Save results
test_results = pd.DataFrame({
    "Filename": test_generator.filenames,
    "Prediction": preds.flatten()
})
test_results["Label"] = test_results["Prediction"].apply(lambda x: "real" if x > 0.5 else "fake")
def emoji_feedback(score):
    if score > 0.9:
        return "🟢 Definitely Real"
    elif score > 0.7:
        return "🟡 Likely Real"
    elif score > 0.3:
        return "🟠 Uncertain"
    else:
        return "🔴 Likely Fake"

test_results["Feedback"] = test_results["Prediction"].apply(emoji_feedback)
test_results.to_csv("prediction_results.csv", index=False)
print(test_results)