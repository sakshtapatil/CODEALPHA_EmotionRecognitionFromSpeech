import numpy as np
import joblib

from tensorflow.keras.models import load_model
from utils import extract_features
model = load_model("models/emotion_model.keras")
label_encoder = joblib.load("models/label_encoder.pkl")

print("===================================")
print("Model Loaded Successfully!")
print("===================================")

audio_path = "dataset/Actor_01/03-01-03-01-01-01-01.wav"
mfcc = extract_features(audio_path)

print("Audio Loaded Successfully!")
print("Input Shape :", mfcc.shape)


prediction = model.predict(mfcc, verbose=0)

predicted_class = np.argmax(prediction)

predicted_emotion = label_encoder.inverse_transform([predicted_class])[0]



print("\n===================================")
print("Prediction Completed")
print("===================================")

print(f"Predicted Emotion : {predicted_emotion}")

print("\nConfidence Scores:\n")

for i, score in enumerate(prediction[0]):

    emotion = label_encoder.inverse_transform([i])[0]

    print(f"{emotion:<12} : {score * 100:.2f}%")