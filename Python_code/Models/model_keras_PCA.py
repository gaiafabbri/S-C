from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta #other choices

'''----------------- MODEL DEFINITION------------------'''
''' The model chosen is a 1D CNN since the images are now represented by one-dimensional arrays after being flattened and reduced by the PCA process
1) The activation function relu is chosen
2) The input shape is given by (number of principal components, batch size) where the number of principal components is previously computed
3) A convolutional layer is inserted with ReLU activation function
4)the MaxPooling layer reduces the input spatial dimension
5) the Flatten layer ensures that the output of the previous steps is a one-dimensional array to be passed to the following dense layer with a relu activation function
6)the output layer uses a sigmoid function to generate the probability distribution for an image to belong to the singal or background classes; this is common for binary classification problem
 '''

def keras_PCA(n_principal_components):
    # Definition and compilation of the model
    model = Sequential([
        Input(shape=(n_principal_components, 1)),
        Conv1D(10, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2, strides=2),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(1, activation='sigmoid')  
    ]) #vedere sigmoid o softmax

    model.compile(optimizer='adam',
              loss='binary_crossentropy',  
              metrics=['accuracy'])
    model.summary()
    return model
