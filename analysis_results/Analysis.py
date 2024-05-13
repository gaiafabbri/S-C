import uproot
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
import os
from DataPreparation.Data_Preparation import load_and_normalize_images, apply_reshape
from numpy.fft import fftshift, fft2

'''------------------ FUNCTIONS DEFINITIONS ------------------'''

# Visualisation of some images
def plot_images(num_images_to_plot, signal_data, background_data):
    fig, axs = plt.subplots(num_images_to_plot, 2, figsize=(10, 12))
    for i in range(num_images_to_plot):
        axs[i, 0].imshow(np.reshape(signal_data[i], (16, 16)))  # Imaging reshaping
        axs[i, 0].set_title('Signal Image')
        axs[i, 1].imshow(np.reshape(background_data[i], (16, 16)))  # Imaging reshaping
        axs[i, 1].set_title('Background Image')
    plt.savefig(os.path.join(results_folder, "images.png"))
    plt.show()
    
# Visualisation of the pixel intensity distribution
def pixel_intensity_distribution (sgn_mean,bkg_mean):
    plt.figure(figsize=(10, 5))
    plt.hist(sgn_mean, bins=30, alpha=0.5, label='Signal Mean', color='blue')
    plt.hist(bkg_mean, bins=30, alpha=0.5, label='Background Mean', color='red')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Pixel Intensity Distribution')
    plt.legend()
    plt.savefig(os.path.join(results_folder, "pixel_intensity_distribution.png"))
    plt.show()
    
# Visualisation of clustering results
def plot_cluster_histogram(cluster1, cluster2):
    plt.figure(figsize=(10, 8))
    plt.hist(cluster1, bins=np.arange(len(np.unique(cluster1)) + 1) - 0.5, color='red', rwidth=0.8, label='signal')
    plt.hist(cluster2, bins=np.arange(len(np.unique(cluster2)) + 1) - 0.5, color='blue', rwidth=0.8, label='background')
    plt.xlabel('Cluster')
    plt.ylabel('Numero di Immagini')
    plt.title('Background and signal cluster')
    plt.grid(True)
    plt.savefig(os.path.join(results_folder, "plot_cluster_histogram.png"))
    plt.show()

#Visualization of cluster centroids
def plot_cluster_centers(centers1, centers2):
    num_clusters1 = len(centers1)
    num_clusters2 = len(centers2)
    
    fig, axs = plt.subplots(2, max(num_clusters1, num_clusters2), figsize=(15, 8))
    
    # Centroid plots of cluster 1
    for i in range(num_clusters1):
        axs[0, i].imshow(np.reshape(centers1[i], (16, 16)), cmap='gray')
        axs[0, i].set_title('Signal'.format(i))
        axs[0, i].axis('off')
    
    # Centroid plot of cluster 2
    for i in range(num_clusters2):
        axs[1, i].imshow(np.reshape(centers2[i], (16, 16)), cmap='gray')
        axs[1, i].set_title('Background'.format(i))
        axs[1, i].axis('off')
    
    plt.suptitle('Cluster centroids')
    plt.savefig(os.path.join(results_folder, "plot_cluster_centers.png"))
    plt.show()
    
#Visualisation of pixel distribution and pixel correlation
def plot_pixel_distribution(signal_image, background_image):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(signal_image.flatten(), bins=50, color='blue', alpha=0.7, label='Signal')
    plt.hist(background_image.flatten(), bins=50, color='red', alpha=0.7, label='Background')
    plt.title('Pixel Distribution')
    plt.xlabel('Brightness')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(signal_image.flatten(), background_image.flatten(), color='green', alpha=0.5)
    plt.title('Pixel Correlation')
    plt.xlabel('Signal Brightness')
    plt.ylabel('Background Brightness')
    
    plt.savefig(os.path.join(results_folder, "plot_pixel_distribution.png"))
    plt.show()
    
#Visualisation of the intensity profile
def plot_intensity_profile(image_data1,image_data2, axis='row'):
    # Fai il reshaping dell'array in un'immagine 2D se necessario
    if len(image_data1.shape) == 1:
        # Calcola le dimensioni dell'immagine
        # Fai il reshaping in un'immagine 2D
        print(image_data1.shape)
        print(image_data2.shape)
        image_data1 = image_data1.reshape((height, width))
        image_data2=image_data2.reshape((height,width))
    
    # Calcola il profilo di intensità medio lungo l'asse specificato
    if axis == 'row':
        intensity_profile1 = np.mean(image_data1, axis=1)
        intensity_profile2 = np.mean(image_data2, axis=1)
        
        ax2 = plt.subplot(1, 2, 1)
        
        ax2.plot(intensity_profile1, label='Signal', color='blue')
        ax2.plot(intensity_profile2, label='Background', color='red')
        ax2.set_xlabel(axis.capitalize() + " Index")
        ax2.set_ylabel("Average Intensity")
        ax2.set_title("Average Intensity Profile along " + axis.capitalize())
        #ax2.show()
        
    elif axis == 'column':
        intensity_profile1 = np.mean(image_data1, axis=0)
        intensity_profile2 = np.mean(image_data2, axis=0)
        
        ax2 = plt.subplot(1, 2, 2)
        
        ax2.plot(intensity_profile1, label='Signal', color='blue')
        ax2.plot(intensity_profile2, label='Background', color='red')
        ax2.set_xlabel(axis.capitalize() + " Index")
        ax2.set_ylabel("Average Intensity")
        ax2.set_title("Average Intensity Profile along " + axis.capitalize())
        #ax2.show()
    else:
        raise ValueError("Axis must be 'row' or 'column'")

'''------------------ DATA LOADING & VARIABLES EXTRACTION ------------------'''

folder_path = "images"
file_names = ["images_data_16x16_10000.root", "images_data_16x16_100000.root"]

existing_file = None

for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    if os.path.exists(file_path):
        existing_file = file_name
        break

if existing_file:
    print("The file", existing_file, "is present in the folder.")
else:
    print("Neither of the two files is present in the folder.")
    
results_folder = "analysis_plots"
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
    
# Estrazione delle dimensioni
dimensions_str = file_name.split("_")[2]
dimensions_without_extension = dimensions_str.split(".")[0]
height, width = map(int, dimensions_without_extension.split("x"))

signal_images, background_images = load_and_normalize_images(folder_path, file_name)

signal_data=np.array(signal_images)
background_data=np.array(background_images)
print("Signal images and background images dimension: ", signal_data.shape,background_data.shape)

signal_data_reshaped=apply_reshape(signal_data, height, width)
background_data_reshaped=apply_reshape(background_data, height, width)
print("Signal images and background images dimension after reshaoing: ", signal_data_reshaped.shape, background_data_reshaped.shape)

'''------------------ STATISTICAL ANALYSIS ------------------'''
# Analisi Statistiche
sgn_mean = np.mean(signal_data, axis=0)
bkg_mean = np.mean(background_data, axis=0)
sgn_std = np.std(signal_data, axis=0)
bkg_std = np.std(background_data, axis=0)

'''------------------ CLUSTERING ALGORITHMS ------------------'''

 #Segmentazione delle Immagini con KMeans
num_clusters = 3  # Modifica il numero di cluster
kmeans_sgn = KMeans(n_clusters=num_clusters, max_iter=300, random_state=42)  # Modifica i parametri di KMeans
sgn_clusters = kmeans_sgn.fit_predict(signal_data)
kmeans_bkg = KMeans(n_clusters=num_clusters, max_iter=300, random_state=42)  # Modifica i parametri di KMeans
bkg_clusters = kmeans_bkg.fit_predict(background_data)

'''------------------ IMAGES ANALYSIS ------------------'''

plot_images(5, signal_data, background_data)

pixel_intensity_distribution(sgn_mean, bkg_mean)

plot_cluster_histogram(sgn_clusters, bkg_clusters)

plot_cluster_centers(kmeans_sgn.cluster_centers_, kmeans_bkg.cluster_centers_,)

plot_pixel_distribution(signal_data_reshaped, background_data_reshaped)

#Figure creation
fig = plt.figure(figsize=(15, 5))
plot_intensity_profile(signal_data,background_data, axis='row')
plot_intensity_profile(signal_data, background_data, axis='column')

plt.savefig(os.path.join(results_folder, "plot_intensity_profile.png"))
plt.tight_layout()
plt.show()


