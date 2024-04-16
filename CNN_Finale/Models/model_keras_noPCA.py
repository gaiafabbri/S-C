from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense

'''------------------ MODEL DEFINITION ------------------'''
'''The model is chosen to be a 2D CNN to deal with the shape of the data, which are images with a dimension width*height
1) The relu activation function is chosen and the padding="same" means that the input and the output have the same dimensions; "kernel initializer" is used to set the weight randomly
2)the input shape is given by (width, height, 1)
3) the BatchNormalization layer is used to make the training more reliable and faster
4) the MaxPooling layer reduces the input spatial dimension
5) the Flatten layer ensures that the output of the previous steps is a one-dimensional array to be passed to the following dense layer with a relu activation function
6)the output layer uses a sigmoid function to generate the probability distribution for an image to belong to the singal or background classes; this is common for binary classification problem
'''

def keras_noPCA(width, height):
    model = Sequential([
        Conv2D(10, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='glorot_normal', input_shape=(width, height, 1)),
        BatchNormalization(),
        Conv2D(10, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='glorot_normal'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=(1, 1)),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model
