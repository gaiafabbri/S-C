import os
import uproot
import numpy as np

def load_and_normalize_images(folder_path, file_name):
    # Ottenere il percorso completo della cartella images
    images_folder_path = os.path.join(os.path.dirname(__file__), folder_path)
    file_path = os.path.join(folder_path, file_name)
    # Aprire il file del dataset
    file = uproot.open(file_path)

    # Caricamento dei dati del segnale
    signal_tree = file["sig_tree"]
    signal_arrays = signal_tree.arrays()
    signal_data = np.array(signal_arrays["vars"])

    # Caricamento dei dati dello sfondo
    background_tree = file["bkg_tree"]
    background_arrays = background_tree.arrays()
    background_data = np.array(background_arrays["vars"])

    # Calcolo del valore massimo dei pixel per le immagini di segnale e sfondo
    max_pixel_values_signal = [np.max(image_s) for image_s in signal_data]
    max_pixel_values_bkg = [np.max(image_b) for image_b in background_data]
    
    '''------------------ DATA NORMALIZATION ------------------'''
    '''
    data need to be normalized between 0 and 1:
    1) the maximum pixel value is calculated both for the signal and the background images
    2) each image is divided for the maximum pixel value
    3) in order to reduce the computational effort, a batch_size=1000 is defined: the loop is repeated for a number of times given by the number of images divided by batch_size
    '''

    max_pixel_values_signal = np.max(max_pixel_values_signal)
    max_pixel_values_bkg = np.max(max_pixel_values_bkg)

    # Normalizzazione delle immagini tra 0 e 1
    signal_images_normalized = [image_s / max_pixel_values_signal for image_s in signal_data]
    background_images_normalized = [image_b / max_pixel_values_bkg for image_b in background_data]

    return signal_images_normalized, background_images_normalized
