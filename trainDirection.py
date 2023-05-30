import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

# Define data path
data_path = f'C:\\Users\\ABENEZER\\Documents\\repos\\Personal\\Directions-CNN\\dataSet\\'

# Define labels and number of classes
labels = ['Kegne', 'Gra', 'Fit', 'Huala', 'Kum']
num_classes = len(labels)

mfccs_list = []  # store mfccs
y = []  # labels will go here

for i, label in enumerate(labels):
    label_dir = os.path.join(data_path, label)
    for filename in os.listdir(label_dir):
        if filename.endswith('.wav'):
            audio_path = os.path.join(label_dir, filename)
            sound, sample_rate = librosa.load(audio_path)
            mfcc = librosa.feature.mfcc(y=sound, sr=sample_rate)
            mfccs_list.append(mfcc)
            y.append(i)  # assign label index

# Pad the sequences to have the same length
max_len = max(mfcc.shape[1] for mfcc in mfccs_list)
X = np.zeros((len(mfccs_list), mfccs_list[0].shape[0], max_len))

for i, mfcc in enumerate(mfccs_list):
    X[i, :, :mfcc.shape[1]] = mfcc
# Convert list to numpy array
X = np.array(X)
y = np.array(y)

# One-hot encode labels
y = to_categorical(y, num_classes=num_classes)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape to fit the network input (samples, bands, frames, channels)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Build CNN model
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=2, input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))  # output layer

# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test), verbose=1)

# Evaluate
score = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: ", score[1])
model.save('directions.h5')

