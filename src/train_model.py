import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout

X = np.load("dataset/X.npy")
y = np.load("dataset/y.npy")

print("=" * 40)
print("Dataset Loaded Successfully")
print("=" * 40)

print("X Shape :", X.shape)
print("y Shape :", y.shape)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("\nEncoded Labels:")
print(y[:10])

# Save Label Encoder
joblib.dump(label_encoder, "models/label_encoder.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain-Test Split Completed")
print("Training Data :", X_train.shape)
print("Testing Data  :", X_test.shape)


X_train = X_train.reshape(X_train.shape[0], 13, 300, 1)
X_test = X_test.reshape(X_test.shape[0], 13, 300, 1)

print("\nAfter Reshaping")
print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)



model = Sequential()

model.add(
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=(13, 300, 1)
    )
)

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(
    Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu"
    )
)

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(8, activation="softmax"))

print("\n")
model.summary()


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)



print("\nTraining Started...\n")

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test)
)



loss, accuracy = model.evaluate(X_test, y_test)

print("\n" + "=" * 40)
print("Model Evaluation")
print("=" * 40)

print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy * 100:.2f}%")


model.save("models/emotion_model.keras")

print("\n" + "=" * 40)
print("Model Saved Successfully!")
print("Location : models/emotion_model.keras")
print("Label Encoder Saved : models/label_encoder.pkl")
print("=" * 40)