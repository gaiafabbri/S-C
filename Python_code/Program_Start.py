import os
import tensorflow as tf
from DataPreparation.Data_Preparation import load_and_normalize_images
from DataPreparation.reshaping import apply_reshape
from DataPreparation.reshaping import create_features_labels
from DataPreparation.shuffle import apply_shuffle
from DataPreparation.PCA import find_optimal_num_components, apply_pca
import numpy as np
from Models.model_keras_noPCA import keras_noPCA
from Models.model_keras_PCA import keras_PCA
from Models.model_DNN import DNN_model
from Models.model_DNN_PCA import DNN_model_PCA
from Models.model_Torch_noPCA import torch_noPCA
from Models.model_Torch_PCA import torch_PCA
from Models.model_BDT import BDT_model
from Evaluation.PredictionNN import eval_Neural_Networks, print_ROC, plot_training_curves_KERAS
from Evaluation.TrainTorch import trained_model, test_eval, torch_eval, plot_training_curves_TORCH
from Evaluation.PredictionBDT import BDT_eval, print_ROC_BDT, plot_training_curves_BDT
from sklearn.model_selection import train_test_split
import time
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from torch.optim import Adam
import matplotlib.pyplot as plt

#Setting the environment to use the CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



'''------------------ DATA LOADING & NORMALIZATION ------------------'''

folder_path = "images"
file_name = "images_data_16x16_100000.root"

# Estrazione delle dimensioni
dimensions_str = file_name.split("_")[2]
dimensions_without_extension = dimensions_str.split(".")[0]
height, width = map(int, dimensions_without_extension.split("x"))
event_number = int(file_name.split("_")[-1].split(".")[0])

signal_images, background_images = load_and_normalize_images(folder_path, file_name)

print("Lunghezza immagini di segnale:", len(signal_images))
print("Lunghezza immagini di segnale:", len(signal_images))

'''------------------ DATA PREPARATION ------------------'''

signal_immages_numpy=np.array(signal_images)
background_images_numpy=np.array(background_images)
print("Signal images and background images dimension: ", signal_immages_numpy.shape, background_images_numpy.shape)

# defining the features and the labels
X,y=create_features_labels(signal_immages_numpy, background_images_numpy)

'''------------------ MODEL CHOICE ------------------'''

available_options = {
    "1": "CNN with torch",
    "2": "CNN with tensorflow-keras",
    "3": "BDT",
    "4": "DNN",
    "5": "Plot comparative ROC and exit"
}

while True:
# Chiedi all'utente di scegliere un'opzione
    print("Choose an option:")
    for key, value in available_options.items():
        print(key, "=", value)
    choice = input("Enter the number corresponding to the desired option:")
    
    
    if choice not in available_options:
        print("Option already explored or invalid. Please enter a number from the remainder.")
        continue
    print("You choose", available_options[choice])
    
    # Leggi l'input dell'utente
    #scelta = input("Inserisci il numero corrispondente all'opzione desiderata: ")
    # Imposta il numero di thread che TensorFlow utilizzerà
    #tf.config.threading.set_intra_op_parallelism_threads(4)
    #tf.config.threading.set_inter_op_parallelism_threads(4)
    # Verifica se l'utente vuole uscire


    if choice == "1": ###SISTEMARE FUNZIONE DI TRAINING
    
        '''------------------ CNN TORCH ------------------'''

        
        '''------------------ PCA APPLICATION ------------------'''

        if event_number > 5000:
            # Fai qualcosa se event_number è maggiore di 5000
            print("Event number is greater than 5000:", event_number)
            # Definisci la varianza desiderata
            target_variance_explained = 0.95
            
            #computing the number of principal components
            n_principal_components=find_optimal_num_components(X,0.95, width, height)
            print(n_principal_components)
            
            # data shuffle lo facciamo dopo?
            X_PCA = apply_pca(X, n_principal_components)

            # Creating PyTorch tensors for data and labels
            X_tensor = torch.tensor(X_PCA, dtype=torch.float32)
            y_tensor = torch.tensor(y) # creation of label 0 = signal, while 1 = background
            # Reshaping of tensor y to match x tensor dimension
            y_tensor = y_tensor.view(-1, 1)
            # Creation of TensorDataset for data and labels
            dataset = TensorDataset(X_tensor, y_tensor)
            # Creation of DataLoader for training set and test set
            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
            print("Dimensioni del training set:")
            for inputs, labels in train_loader:
                print("X_train:", inputs.shape)
                print("y_train:", labels.shape)
                break

            print("\nDimensioni del test set:")
            for inputs, labels in test_loader:
                print("X_test:", inputs.shape)
                print("y_test:", labels.shape)
                break

            #model definition
            model = torch_PCA(width, height, n_principal_components)
            # Define hyperparameters
            num_epochs = 10
            batch_size = 64
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam
            start_time=time.time() #monitoring the starting time
        
            # Addestramento del modello
            model, history = trained_model(model, train_loader, test_loader, num_epochs, batch_size, optimizer, criterion, save_best=None, scheduler=None)
            criterion = nn.BCELoss().to(torch.float32)  # Specifies the data type of the loss function

            end_time=time.time() #monitoring the finish time
            training_time = end_time - start_time  #q Calcola il tempo totale di addestramento
        
            print(f"Tempo di addestramento: {training_time} secondi")
            
            pass
            
            '''------------------ NO NEED FOR PCA ------------------'''

        else:
            print("Event number is less than or equal to 5000:", event_number)
            
            #data preparation
            # Creating PyTorch tensors for data and labels
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y) # creation of label 0 = signal, while 1 = background
            # Reshaping of tensor y to match x tensor dimension
            y_tensor = y_tensor.view(-1, 1)
            # Creation of TensorDataset for data and labels
            dataset = TensorDataset(X_tensor, y_tensor)
            # Creation of DataLoader for training set and test set
            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            
            model = torch_noPCA(width, height)
        
            # Define hyperparameters
            num_epochs = 10
            batch_size = 64
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam
            
            start_time=time.time() #monitoring the starting time

            #  Train the model
            model, history = trained_model(model, train_loader, test_loader, num_epochs, batch_size, optimizer, criterion, save_best=None, scheduler=None)

            criterion = nn.BCELoss().to(torch.float32)  # Specifies the data type of the loss function

            end_time=time.time() #monitoring the finish time
            training_time = end_time - start_time  # Calcola il tempo totale di addestramento
    
            print(f"Tempo di addestramento: {training_time} secondi")

            pass
            
        test_eval(model, test_loader, criterion)
        # Creazione della figura
        fig = plt.figure(figsize=(15, 5))
        true_labels, y_pred, y_pred_classes=torch_eval(model, test_loader)
        # Chiamata alla funzione per il plot della ROC curve
        background_rejection_CNN_torch, tpr_CNN_torch = print_ROC(true_labels, y_pred, y_pred_classes)
        plot_training_curves_TORCH(history)
        

        
    elif choice == "2":
    
        '''------------------ CNN KERAS ------------------'''
        
        # Setting the number of threads used by TensorFlow for intra-op and inter-op operations
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(4)
        
        
        '''------------------ PCA APPLICATION ------------------'''
        
        if event_number > 5000: #POTREMMO FARE LA PCA DOPO LA CONCATENAZIONE!!!
            # Fai qualcosa se event_number è maggiore di 5000
            print("Event number is greater than 5000:", event_number)
            # Definisci la varianza desiderata
            target_variance_explained = 0.95
        
            n_principal_components=find_optimal_num_components(X,0.95, width, height)
            print(n_principal_components)
            
            # data shuffle
            X_shuffled=apply_shuffle(X)
            y_shuffled=apply_shuffle(y)
    
            X_PCA = apply_pca(X_shuffled, n_principal_components)
        
            # Splitting the dataset in 80% training and 20% testing
            X_train, X_test, y_train, y_test = train_test_split(X_PCA, y_shuffled, test_size=0.2, random_state=42)
        
            print("Training set dimension:")
            print("X_train:", X_train.shape)
            print("y_train:", y_train.shape)
            print("\nTesting set dimensions:")
            print("X_test:", X_test.shape)
            print("y_test:", y_test.shape)
    
            model = keras_PCA(n_principal_components)
            start_time = time.time() #setting the time when the training starts
            history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
            end_time = time.time() #monitoring the time when the trainign ends

            training_time_seconds = end_time - start_time
            print("Training time with the CPU:", training_time_seconds, "s")
            
        else:
            '''------------------ NO NEED FOR PCA ------------------'''
            print("Event number is less than or equal to 5000:", event_number)
            X_reshaped=apply_reshape(X, width, height)
    
            # data shuffle
            X_shuffled=apply_shuffle(X_reshaped)
            y_shuffled=apply_shuffle(y)
            
            # Splitting the dataset in 80% training and 20% testing
            X_train, X_test, y_train, y_test = train_test_split(X_shuffled, y_shuffled, test_size=0.2, random_state=42)

            print("Training set dimension:")
            print("X_train:", X_train.shape)
            print("y_train:", y_train.shape)
            print("\nTesting set dimensions:")
            print("X_test:", X_test.shape)
            print("y_test:", y_test.shape)
            # Calcolo del numero totale di dati nel set di addestramento dopo l'aumentazione

            model = keras_noPCA(width, height)
            start_time = time.time() #setting the time when the training starts
            history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
            end_time = time.time() #monitoring the time when the trainign ends

            training_time_seconds = end_time - start_time
            print("Training time with the CPU:", training_time_seconds, "s")
            
        
            pass
            
        # Creazione della figura
        fig = plt.figure(figsize=(15, 5))
        y_pred, y_pred_classes=eval_Neural_Networks(model, X_test, y_test)
        # Chiamata alla funzione per il plot della ROC curve
        background_rejection_CNN_keras, tpr_CNN_keras = print_ROC(y_test, y_pred, y_pred_classes)
        # Chiamata alla funzione per il plot delle curve di addestramento
        plot_training_curves_KERAS(history)

        
        
    elif choice == "3":
        '''------------------ BDT ------------------'''
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Training set dimensions:")
        print("X_train:", X_train.shape)
        print("y_train:", y_train.shape)
        print("\nTesting set dimensions:")
        print("X_test:", X_test.shape)
        print("y_test:", y_test.shape)

        
        # Definizione e addestramento del modello xgboost
        start_time = time.time() #setting the time when the training starts
        model, evals_result = BDT_model(X_train, X_test, y_train, y_test)


        end_time = time.time() #monitoring the time when the trainign ends
        training_time_seconds = end_time - start_time
        print("Training time with the CPU:", training_time_seconds, "s")

        y_pred, y_pred_binary = BDT_eval(model, X_test, y_test)
        
        # Creazione della figura
        fig = plt.subplots(figsize=(15, 5))

        # Chiamata alla funzione per il plot della ROC curve
        background_rejection_BDT, tpr_BDT= print_ROC_BDT(y_test, y_pred, y_pred_binary)
        # Chiamata alla funzione per il plot delle curve di addestramento
        plot_training_curves_BDT(evals_result)


        
        pass
        
    elif choice == "4":
        # Setting the number of threads used by TensorFlow for intra-op and inter-op operations
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(4)
        
        
        if event_number > 100001:
        
            '''------------------ PCA APPLICATION ------------------'''
            # Fai qualcosa se event_number è maggiore di 5000
            print("Event number is greater than 5000:", event_number)
            # Definisci la varianza desiderata
            target_variance_explained = 0.95
        
            n_principal_components=find_optimal_num_components(X,0.95, width, height)
            print(n_principal_components)
            
            # data shuffle
            X_shuffled=apply_shuffle(X)
            y_shuffled=apply_shuffle(y)
    
            X_PCA = apply_pca(X_shuffled, n_principal_components)
        
            # Splitting the dataset in 80% training and 20% testing
            X_train, X_test, y_train, y_test = train_test_split(X_PCA, y_shuffled, test_size=0.2, random_state=42)
        
            print("Training set dimension:")
            print("X_train:", X_train.shape)
            print("y_train:", y_train.shape)
            print("\nTesting set dimensions:")
            print("X_test:", X_test.shape)
            print("y_test:", y_test.shape)
            
            model = DNN_model_PCA(n_principal_components)  # Creazione del modello
            # Training del modello
            start_time = time.time()  # Imposta il tempo di inizio del training
            history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
            end_time = time.time()  # Monitora il tempo di fine del training
            training_time_seconds = end_time - start_time
            print("Training time with the CPU:", training_time_seconds, "s")
            
            y_pred, y_pred_classes=eval_Neural_Networks(model, X_test, y_test)

            pass
        
        else:
        
            '''------------------ NO NEED FOR PCA------------------'''
                
            print("Event number is less than or equal to 5000:", event_number)
                
            # data shuffle
            X_shuffled=apply_shuffle(X)
            y_shuffled=apply_shuffle(y)
            
            # Splitting the dataset in 80% training and 20% testing
            X_train, X_test, y_train, y_test = train_test_split(X_shuffled, y_shuffled, test_size=0.2, random_state=42)
        
            print("Training set dimension:")
            print("X_train:", X_train.shape)
            print("y_train:", y_train.shape)
            print("\nTesting set dimensions:")
            print("X_test:", X_test.shape)
            print("y_test:", y_test.shape)
            
            
            model = DNN_model(width, height)  # Creazione del modello
            # Training del modello
            start_time = time.time()  # Imposta il tempo di inizio del training
            history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
            end_time = time.time()  # Monitora il tempo di fine del training
            training_time_seconds = end_time - start_time
            print("Training time with the CPU:", training_time_seconds, "s")
            
            y_pred, y_pred_classes=eval_Neural_Networks(model, X_test, y_test)
            pass
            
        # Creazione della figura
        fig = plt.figure(figsize=(15, 5))

        # Chiamata alla funzione per il plot della ROC curve
        background_rejection_DNN, tpr_DNN = print_ROC(y_test, y_pred, y_pred_classes)
        

        # Chiamata alla funzione per il plot delle curve di addestramento
        plot_training_curves_KERAS(history)
        
    elif choice == "5":
        
        # Creazione di un dizionario per memorizzare i risultati delle curve ROC per ciascun modello
        roc_results = {}

        # Aggiungi i risultati della curva ROC per il modello CNN Torch, se disponibili
        if 'background_rejection_CNN_torch' in locals() and 'tpr_CNN_torch' in locals():
            roc_results['CNN Torch'] = (background_rejection_CNN_torch, tpr_CNN_torch)

        # Aggiungi i risultati della curva ROC per il modello CNN Keras, se disponibili
        if 'background_rejection_CNN_keras' in locals() and 'tpr_CNN_keras' in locals():
            roc_results['CNN Keras'] = (background_rejection_CNN_keras, tpr_CNN_keras)

        # Aggiungi i risultati della curva ROC per il modello BDT, se disponibili
        if 'background_rejection_BDT' in locals() and 'tpr_BDT' in locals():
            roc_results['BDT'] = (background_rejection_BDT, tpr_BDT)

        # Aggiungi i risultati della curva ROC per il modello DNN, se disponibili
        if 'background_rejection_DNN' in locals() and 'tpr_DNN' in locals():
            roc_results['DNN'] = (background_rejection_DNN, tpr_DNN)

        # Creazione della figura
        fig = plt.figure(figsize=(7,5))

        # Plot delle curve ROC per i modelli disponibili
        for model_name, (background_rejection, tpr) in roc_results.items():
            plt.plot(background_rejection, tpr, label=model_name)

        # Imposta etichette e titolo
        plt.xlabel('Background Rejection')
        plt.ylabel('Signal Efficiency')
        plt.title('Receiver Operating Characteristic (ROC) Curve for all Models')
        plt.plot([1, 0], [0, 1], color='navy', lw=2, linestyle='--')
        # Aggiungi legenda
        plt.legend()
        plt.savefig(f"plot_results/Comparison_among_models_{event_number}_{width}x{height}.png")
        # Mostra la figura
        plt.tight_layout()
        plt.show()

        
        break

    plt.savefig(f"plot_results/{available_options[choice]}_{event_number}_{width}x{height}.png")
    plt.tight_layout()
    plt.show()
    del available_options[choice]
    
#model_keras.summary()
#model.load_state_dict(best_model_state) vedere se fare cose tipo questo, sono cagatine carine
