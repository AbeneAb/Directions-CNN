import pyaudio
import numpy as np
import python_speech_features as psf
from tensorflow.keras.models import load_model
import threading

# Load model
model = load_model('directions.h5')

# Constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050  # Match with the rate we used earlier
CHUNK = 16000  # 1 second of audio

# Map index to corresponding label
labels = ['Kegne', 'Gra', 'Fit', 'Huala', 'Kum']


def get_mfccs(signal, rate):
    mfccs = psf.mfcc(signal, rate, numcep=13, nfilt=26)
    mfccs = mfccs.T
    if mfccs.shape[1] < 99:
        mfccs = np.pad(
            mfccs, ((0, 0), (0, 99 - mfccs.shape[1])), mode='constant')
    else:
        mfccs = mfccs[:, :99]
    return mfccs


def calculate_energy(signal):
    energy = np.sum(signal ** 2) / len(signal)
    return energy


# Create an audio object
audio = pyaudio.PyAudio()

# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

# Create a flag to indicate recording status
is_recording = True

# Define a function to listen for a stop command


def listen_for_stop():
    global is_recording
    input("Press 'q' then Enter to stop recording: ")
    is_recording = False


# Start the stop listener in a separate thread so it doesn't block the recording
stop_listener = threading.Thread(target=listen_for_stop)
stop_listener.start()

print("Listening...")
threshold = 3000  # Adjust this value according to your needs


while is_recording:
    # Record audio in chunks and process it
    data = stream.read(CHUNK)
    signal = np.frombuffer(data, dtype=np.int16)

    energy = calculate_energy(signal)
    print(energy)
    # Skip prediction if energy is below a certain threshold
    if energy < threshold:
        continue

    mfccs = get_mfccs(signal, RATE)
    mfccs = mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1)

    try:
        prediction = model.predict(mfccs)
        predicted_index = np.argmax(prediction[0])
        predicted_label = labels[predicted_index]
        prediction_percentage = round(prediction[0][predicted_index] * 100, 2)

        accuracy = model.evaluate(mfccs, np.array([1]))[1] * 100
        if accuracy > 98:
            print("Predicted label: ", predicted_label)
            print("Prediction percentage: ", prediction_percentage, "%")
    except Exception as e:
        print("An error occurred during prediction: ", str(e))


# stop Recording
stream.stop_stream()
stream.close()
audio.terminate()
