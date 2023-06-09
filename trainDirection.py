import os
import numpy as np
from python_speech_features import mfcc
from scipy.io import wavfile
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from pydub import AudioSegment
import random


def augment_audio(audio):
    # Noise addition
    noise = AudioSegment.from_mono_audiosegments(
        AudioSegment.silent(duration=len(audio)))
    for _ in range(len(noise)):
        noise = noise.overlay(audio, position=random.randint(0, len(audio)))
    audio_with_noise = audio.overlay(noise)

    # Time shifting
    shift_audio = audio[1000:] + audio[:1000]

    return [audio_with_noise, shift_audio]


def preprocess_data(data_path, labels):
    y = []  # labels will go here

    mfccs_list = []  # store mfccs

    fixed_shape = (13, 100)

    for i, label in enumerate(labels):
        label_dir = os.path.join(data_path, label)
        for filename in os.listdir(label_dir):
            if filename.endswith('.wav'):
                audio_path = os.path.join(label_dir, filename)
                audio = AudioSegment.from_wav(audio_path)
                audio_arr = np.array(audio.get_array_of_samples())
                mfcc_features = mfcc(audio_arr, samplerate=audio.frame_rate, nfft=1103).T

                # Resize the array to fixed size
                mfcc_resized = zoom(mfcc_features, (fixed_shape[0] / mfcc_features.shape[0], fixed_shape[1] / mfcc_features.shape[1]))

                mfccs_list.append(mfcc_resized)
                y.append(i)  # assign label index

    X = np.array(mfccs_list)
    y = np.array(y)
    y = to_categorical(y, num_classes=num_classes)

    return X, y


def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=2,
              input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'], optimizer='adam')
    return model


# Define data path
data_path = f'C:\\Users\\ABENEZER\\Documents\\repos\\Personal\\Directions-CNN\\dataSet\\'

# Define labels and number of classes
labels = ['Kegne', 'Gra', 'Fit', 'Huala', 'Kum']
num_classes = len(labels)

# Preprocess data
X, y = preprocess_data(data_path, labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Reshape to fit the network input (samples, bands, frames, channels)
X_train = X_train.reshape(
    X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Build and train model
model = build_model(
    (X_train.shape[1], X_train.shape[2], X_train.shape[3]), num_classes)
model.fit(X_train, y_train, batch_size=32, epochs=50,
          validation_data=(X_test, y_test), verbose=1)

score = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: ", score[1])

# Save model
model.save('directions.h5')

print("Model trained and saved as directions.h5")
