import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

def augment_audio(audio, sample_rate):
    # Pitch tuning
    pitch_shift = np.random.uniform(low=-2.5, high=2.5)
    audio_pitch_tuned = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=pitch_shift)

    # Time stretching
    stretch_factor = np.random.uniform(low=0.8, high=1.2)
    audio_time_stretched = librosa.effects.time_stretch(audio, rate=stretch_factor)

    # Add noise
    noise_factor = np.random.uniform(low=0.01, high=0.05)
    white_noise = np.random.randn(len(audio))
    audio_with_noise = audio + noise_factor * white_noise

    # Time shifting
    shift = np.random.randint(sample_rate * 2)
    shift_audio = np.roll(audio, shift)

    return [audio_pitch_tuned, audio_time_stretched, audio_with_noise, shift_audio]



def preprocess_data(data_path, labels):
    mfccs_list = []  # store mfccs
    y = []  # labels will go here

    for i, label in enumerate(labels):
        label_dir = os.path.join(data_path, label)
        for filename in os.listdir(label_dir):
            if filename.endswith('.wav'):
                audio_path = os.path.join(label_dir, filename)
                sound, sample_rate = librosa.load(audio_path)
                augmented_sounds = augment_audio(sound, sample_rate)
                for augmented_sound in augmented_sounds:
                    mfcc = librosa.feature.mfcc(y=augmented_sound, sr=sample_rate)
                    mfccs_list.append(mfcc)
                    y.append(i)  # assign label index

    max_len = max(mfcc.shape[1] for mfcc in mfccs_list)
    X = np.zeros((len(mfccs_list), mfccs_list[0].shape[0], max_len))

    for i, mfcc in enumerate(mfccs_list):
        X[i, :, :mfcc.shape[1]] = mfcc

    y = np.array(y)
    y = to_categorical(y, num_classes=num_classes)

    return X, y

def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=2, input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    return model

# Define data path
data_path = f'C:\\Users\\ABENEZER\\Documents\\repos\\Personal\\Directions-CNN\\dataSet\\'

# Define labels and number of classes
labels = ['Kegne', 'Gra', 'Fit', 'Huala', 'Kum']
num_classes = len(labels)

# Preprocess data
X, y = preprocess_data(data_path, labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape to fit the network input (samples, bands, frames, channels)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Build and train model
model = build_model((X_train.shape[1], X_train.shape[2], X_train.shape[3]), num_classes)
model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test), verbose=1)

score = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: ", score[1])

# Save model
model.save('directions.h5')

print("Model trained and saved as directions.h5")


