import librosa
import numpy as np
from tensorflow.keras.models import load_model


model = load_model('directions.h5')


# Load new .wav file
file_path = 'C:\\Users\\ABENEZER\\Documents\\repos\\Personal\\Directions-CNN\\dataSet\\Gra\\1.wav'
sound, sample_rate = librosa.load(file_path)
max_len = 88
# Extract MFCCs
mfcc = librosa.feature.mfcc(y=sound, sr=sample_rate)

# Pad MFCCs to match the input shape of the model
padded_mfcc = np.zeros((mfcc.shape[0], max_len))
padded_mfcc[:, :mfcc.shape[1]] = mfcc

# Reshape to fit the network input (samples, bands, frames, channels)
padded_mfcc = padded_mfcc.reshape(1, padded_mfcc.shape[0], padded_mfcc.shape[1], 1)

# Make prediction
prediction = model.predict(padded_mfcc)
predicted_index = np.argmax(prediction[0])

# Map index to corresponding label
predicted_label = labels[predicted_index]

print("The predicted label is:", predicted_label)
