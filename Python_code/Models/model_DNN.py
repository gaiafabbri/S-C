from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.initializers import GlorotNormal

'''------------------MODEL DEFINITION-----------------------'''
'''The model is defined as follows:
1)the input layer is defined as (nh*nw,) which are the images dimension
2)A dense layer is added to provide an output as the linear combination od the inputs
3)the normalization is applied to make the traoining more stable and reliable
4)The previous step are repeated to make the model learn complex representation of data
5)A final layer with one neuron and a sigmoid function is chosen since we deal with a binary classification problem: the sigmoid function has an output between 0 and 1, resulting into a probability to have background or signal
6)The loss function 'binary_crossentropy' is suitable for a binary classification problem
'''
def DNN_model(width, height):
    # Defining the model
    model = Sequential()
    
    # Input layer
    model.add(Dense(64, activation='relu', input_shape=(width*height,), kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    
    # Hidden layers
    model.add(Dense(256, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compiling the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout

def DNN_model(width, height):
    # Defining the model
    model = Sequential()
    model.add(Input(shape=(width*height,))) # Input layer
    
    # Hidden layers
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))  # Adding dropout for regularization
    
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compiling the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
'''
