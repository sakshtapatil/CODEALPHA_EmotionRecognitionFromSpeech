import os
import librosa
import numpy as np

dataset_path = "dataset"
emotion_dict = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprised"
}

X = []
y = []

print("===================================")
print("Dataset preparation started...")
print("===================================")

folders = os.listdir(dataset_path)

print(f"Total Actor Folders: {len(folders)}")

for folder in folders:

    folder_path = os.path.join(dataset_path, folder)

    files = os.listdir(folder_path)

    
    for file in files:

        file_path = os.path.join(folder_path, file)

        audio, sample_rate = librosa.load(file_path, sr=None)

        mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=13
         )
        if mfcc.shape[1] > 300:
         mfcc = mfcc[:, :300]
        else:
         padding = 300 - mfcc.shape[1]
         mfcc = np.pad(mfcc, ((0, 0), (0, padding)), mode='constant')

        emotion_code = file.split("-")[2]
        emotion = emotion_dict[emotion_code]
    
        X.append(mfcc)
        y.append(emotion)
        print(f"Loaded: {file} --> {emotion}")

X = np.array(X)
y = np.array(y)

print("\n===================================")
print("Dataset Preparation Completed!")
print("===================================")

print("X Shape:", X.shape)
print("y Shape:", y.shape)

np.save("dataset/X.npy", X)
np.save("dataset/y.npy", y)

print("\nDataset saved successfully!")
