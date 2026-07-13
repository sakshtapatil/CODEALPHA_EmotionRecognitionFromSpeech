import librosa
import librosa.display
import matplotlib.pyplot as plt


audio_path = "dataset/Actor_01/03-01-01-01-01-01-01.wav"


audio, sample_rate = librosa.load(audio_path, sr=None)

print(f"Sample Rate : {sample_rate}")
print(f"Duration    : {len(audio)/sample_rate:.2f} seconds")
plt.figure(figsize=(12, 4))

librosa.display.waveshow(audio, sr=sample_rate)

plt.title("Waveform of Audio")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.grid(True)

plt.show()