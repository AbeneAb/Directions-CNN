# Directions-CNN

MFCC stands for Mel-Frequency Cepstral Coefficients. It's a method of extracting features from audio. It was developed for speech recognition systems, but it's often used for music genre classification, musical instrument recognition, and other audio-related tasks.

The process of calculating MFCC features is complex, but in essence, it's about representing the short-term power spectrum of a sound in a way that de-emphasizes the higher frequencies, which are less important for understanding speech or musical sounds.

MFCCs are calculated by taking the Fourier Transform of a signal (converting it from the time domain to the frequency domain), applying a Mel filterbank to it (this is where the "Mel" in "MFCC" comes from, The Mel scale is a perceptual scale of pitches where equal distances correspond to equal perceptual differences. In other words, it's a scale that reflects how humans hear sounds. The name comes from the word "melody" to indicate that the scale is pitch based), taking the log of the powers at each frequency, and then taking the Discrete Cosine Transform of that list.

The purpose of the Mel filterbank is to mimic the human ear's bandpass filters and create coefficients that can be used to represent the envelope of the time power spectrum. Each filter in the filterbank is applied to the power spectrum of an audio signal (obtained by applying a Fourier transform to the signal) to get one coefficient. By applying all the filters in the filterbank, you get a set of coefficients which can be used as features for an audio signal.

These coefficients (MFCCs) are then often used as the features for machine learning tasksof classification. because they capture the relevant characteristics of the audio signals while discarding the unnecessary details, which makes them good for these kinds of tasks.
The result is a set of "cepstral coefficients" that represent the shape of the power spectrum of the original signal. For our neural network, these coefficients become a kind of "feature" that we can learn from.

Number of Layers in ConvNet

In your ConvNet model, there are following layers:

Conv2D layer: This is the convolutional layer. It applies convolution operation on the input, passing the result to the next layer. The parameter 64 is the number of filters that the convolutional layer will learn, while kernel_size=2 specifies the height and width of the convolution window.

MaxPooling2D layer: Pooling layers reduce the spatial size (width and height) of the input, which helps to decrease the computational power required to process the data through dimensionality reduction. It also helps to extract dominant features.

Dropout layer: This is a regularization technique that helps to prevent overfitting in neural networks. During training, it randomly sets a fraction (0.2 in this case) of input units to 0 at each update.

Flatten layer: This layer collapses the spatial dimensions of the input into the channel dimension. In other words, it converts the 2D matrix (height and width) into a 1D array. This is necessary before passing the data to the fully connected layer.

Dense layer: This is the output layer of the network, also known as the fully connected layer. It performs classification on the features extracted by the convolutional layers and down-sampled by the pooling layers. In this layer, the final classification is done by the softmax activation function.
