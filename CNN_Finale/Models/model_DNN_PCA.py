from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, BatchNormalization


'''------------------MODEL DEFINITION-----------------------'''
'''The model is defined as follows:
1)the input layer is defined as (nh*nw,) which are the images dimension
2)A dense layer is added to provide an output as the linear combination od the inputs
3)the normalization is applied to make the traoining more stable and reliable
4)The previous step are repeated to make the model learn complex representation of data
5)A final layer with one neuron and a sigmoid function is chosen since we deal with a binary classification problem: the sigmoid function has an output between 0 and 1, resulting into a probability to have background or signal
6)The loss function 'binary_crossentropy' is suitable for a binary classification problem
'''

def DNN_model_PCA(num_components):
    # Defining the model
    model_keras = Sequential()
    model_keras.add(Input(shape=(num_components,))) #input layer
    model_keras.add(Dense(100, activation='relu')) #dense layer
    model_keras.add(BatchNormalization()) #normalization layer
    for _ in range(3):
        model_keras.add(Dense(100, activation='relu'))
        model_keras.add(BatchNormalization())
    model_keras.add(Dense(1, activation='sigmoid'))

    #compiling the model
    model_keras.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model_keras
