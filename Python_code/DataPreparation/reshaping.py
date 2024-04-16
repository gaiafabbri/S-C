import numpy as np

'''------------------ DATA PREPARATION ------------------'''
'''
1) signal and background images are labelled with 0 and 1 respectively
2) the dataset is shuffled and spplitted: 80% is used to train, 20% is used to test
'''

#fai una funzione che fa il reshape come è stato fatto sopra
def apply_reshape(numpy_image, width, height):
    reshaped_image = numpy_image.reshape(-1, width, height, 1)
    
    print("Dimension: ",reshaped_image.shape)
    
    return reshaped_image
    
    #fai una funzione che fa la concatenazione come è stato fatto sopra
   # defining the features and the labels
def create_features_labels(signal_data, background_data):
    X = np.concatenate([signal_data, background_data]) #features
    y = np.concatenate([np.zeros(len(signal_data)),  np.ones(len(background_data))]) #labels
    
    return X,y
