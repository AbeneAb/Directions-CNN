import numpy as np
from pydub import AudioSegment
from python_speech_features import mfcc
from tensorflow.keras.models import load_model
from scipy.ndimage import zoom

# Load model
model = load_model('directions.h5')

# Load new .wav file
file_path = 'C:\\Users\\ABENEZER\\Documents\\repos\\Personal\\Directions-CNN\\dataSet\\Fit\\20.wav'
audio = AudioSegment.from_wav(file_path)
audio_arr = np.array(audio.get_array_of_samples())

# Extract MFCCs
mfcc_features = mfcc(audio_arr, samplerate=audio.frame_rate, nfft=1103).T

# Resize the array to fixed size
fixed_shape = (13, 100)  # This should be the same shape as you used while training
mfcc_resized = zoom(mfcc_features, (fixed_shape[0] / mfcc_features.shape[0], fixed_shape[1] / mfcc_features.shape[1]))

# Reshape to fit the network input (samples, bands, frames, channels)
mfcc_reshaped = mfcc_resized.reshape(1, mfcc_resized.shape[0], mfcc_resized.shape[1], 1)

# Make prediction
prediction = model.predict(mfcc_reshaped)
predicted_index = np.argmax(prediction[0])

# Map index to corresponding label
labels = ['Kegne', 'Gra', 'Fit', 'Huala', 'Kum']
predicted_label = labels[predicted_index]

print("The predicted label is:", predicted_label)
