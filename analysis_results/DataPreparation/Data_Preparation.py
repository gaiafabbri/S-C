import os
import uproot
import numpy as np

def load_and_normalize_images(folder_path, file_name):
    # Get the full path to the images folder
    images_folder_path = os.path.join(os.path.dirname(__file__), folder_path)
    file_path = os.path.join(folder_path, file_name)
    # Open file
    file = uproot.open(file_path)

    # Loading signal data
    signal_tree = file["sig_tree"]
    signal_arrays = signal_tree.arrays()
    signal_data = np.array(signal_arrays["vars"])
        
        
    # Loading bkg data
    background_tree = file["bkg_tree"]
    background_arrays = background_tree.arrays()
    background_data = np.array(background_arrays["vars"])
    
    '''------------------ DATA NORMALIZATION ------------------'''
    '''
    data need to be normalized between 0 and 1:
    1) the maximum pixel value is calculated both for the signal and the background images
    2) each image is divided for the maximum pixel value
    '''
    # Calculating the maximum pixel value for signal and background images
    max_pixel_values_signal = [np.max(image_s) for image_s in signal_data]
    max_pixel_values_bkg = [np.max(image_b) for image_b in background_data]

    max_pixel_values_signal = np.max(max_pixel_values_signal)
    max_pixel_values_bkg = np.max(max_pixel_values_bkg)

    # Normalisation of images between 0 and 1
    signal_images_normalized = [image_s / max_pixel_values_signal for image_s in signal_data]
    background_images_normalized = [image_b / max_pixel_values_bkg for image_b in background_data]

    return signal_images_normalized, background_images_normalized
    
# Reshape of the images from a numpy array to a images of dimension (width, height)
def apply_reshape(numpy_image, width, height):
    reshaped_image = numpy_image.reshape(-1, width, height, 1)
    
    print("Dimension: ",reshaped_image.shape)
    
    return reshaped_image
    
