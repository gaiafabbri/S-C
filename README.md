# S-C


# Project description
The following project aims to classify signal and background images using machine learning techniques. Ci sono due file principali per l'addestramento sul dataset (vedi Folders organisation): uno basato su TMVA e l'altro sui framework definiti da Python come Tensorflow-Keras, Torch, BDT. 

in particular, a comparison is made between TMVA (Toolkit for Multi-Variate Analysis) packages and Python libraries. The former includes CNN, DNN and Boosted Decision Tree (BDT); the latter implements the same algorithms: CNN is permormed with both keras-tensorflow and pytorch, BDT uses xgboost classifier, while DNN classification is based on keras-tensorflow methods. The models are evaluated in terms of precision, accuracy, f1-score and roc curve



# Folders organisation 

The project structure includes:

- A folder named "__ROOT_Gen__" that holds a file named "Generation.C". This file is responsible for generating the dataset of images which are saved into "images" subfolder to be used. For more details on dataset generation, refer to the dataset section.
- Another folder named "__CNN_python__", which encompasses several subfolders and generates an additional folder post-code execution: plot_results" which aids in visualizing training parameters alongside the Receiver Operating Characteristic (ROC) curve. This curve illustrates signal efficiency versus background rejection for each model. The "plot_results" folder also contains a comparative visualization of ROC curves and a table showcasing performance metrics such as f1 score, accuracy, precision, and training time across different models.
    - There's "_Program_Start.py_" which serves as the main execution script for machine learning methods. These methods include Convolutional Neural Networks (CNN) implemented with both Torch and TensorFlow-Keras, as well as Boosted Decision Trees (BDT) and Deep Neural Networks (DNN) with Keras. The script acts as an interface with the user, orchestrating functions imported from Python scripts in the subfolders to create a unified training file according to user-selected models. Detailed explanations are provided within the code comments.
    - The "_DataPreparation_" subfolder houses scripts for data preparation, including:
        - "Data_Preparation.py" for loading data into arrays and subsequent normalization.
        - "reshaping.py" for reshaping data, particularly useful when training involves image data (2D).
        - "PCA.py" for data preparation when training involves numpy arrays (1D).
        - "shuffle.py" for shuffling dataset entries.
    - The "_Models_" subfolder contains various model implementations, each available in both 1D (after Principal Component Analysis, PCA) and 2D versions (without PCA). Notably, the BDT implementation is an exception. Files within this subfolder include:
         - "model_BDT.py"
         - "model_DNN_PCA.py"
         - "model_DNN.py"
         - "model_keras_noPCA.py"
         - "model_keras_PCA.py"
         - "model_Torch_noPCA.py"
         - "model_Torch_PCA.py"
      - Lastly, the "_Evaluation_" subfolder hosts files for assessing model performance, including:
        - "PredictionNN.py", designed for both CNNs implemented with Keras-TensorFlow.
        - "PredictionBDT.py"
        - "TrainTorch.py"
        - "Table.py", responsible for tabulating results from each model (f1 score, accuracy, precision), highlighting the best-performing model. Additionally, it includes a column displaying the training time for each investigated model.
- Inside the project, there's also a folder named "__TMVA_ML__". This folder contains a file named "TMVA.C". When executed with the provided bash file (see below), it will generate an additional folder named "images". This folder is generated using "Generation.C" and contains the dataset created. The methods implemented within "TMVA.C" include a Convolutional Neural Network (CNN), Deep Neural Network (DNN), and Boosted Decision Tree (BDT). This file originates from an example provided in the ROOT tutorials, accessible via the following link: 'https://root.cern/doc/master/TMVA__CNN__Classification_8C.html'

- An executable bash file called '__Script.sh__' which serves as a tool for project setup and management. It automates various tasks such as project cleanup, verification of ROOT and Python installations, library compatibility checks, and dataset generation. It allows users to select their preferred training environment (ROOT or Python).
- Il "__dockerfile__" che è stato costruito nel caso in cui l'utente non abbia ROOT e/o Python. Esso crea un ambiente per poter usare sia ROOT, sia Python indipendentemente dalla presenza di uno o dell'altro sul computer. In termini pratici sostituirà l'eseguibile "run.sh" replicandone i compiti e la gestione del progetto.


# Data Generation
Il dataset viene generato utilizzando ROOT (vedi Folders organisation) e chiedendo all'utente se usare un file generato di default 16x16 con 100'000 eventi per segnale e per rumore, oppure se generarlo lui, e in quel caso l'utente avrà il libero arbitrio di generare il dataset delle dimensioni che vorrà e con immagini del formato che desidera, anche non quadrata. __COMMENTARE__


## Some comments about dimension of images
The code has been tested on the default dataset but adapted to cover a general scenario. Regarding the analysis conducted in Python, two preprocessing strategies are explored before training: with PCA or without PCA. The idea is that models could be either 1D or 2D, specifically in reference to Convolutional Neural Networks (CNN) using Keras-TensorFlow and Torch, as well as Deep Neural Networks (DNN). This depends on the dimensions of the dataset because, obviously, if we have 2x2 images, PCA will not be applied.

A threshold is established around the image dimensions, such as 12x12 or any combination AxB where both A and B are less than 12 or thei product is lower than 144. If the images width and height product exceeds 144, PCA will be applied to enable training with 1D models, otherwise a 2D model is considered.

## Some comments about dimension of dataset

The dataset has been tested using 100'000 images for the signal and 100'000 for the background, producing consistent results with relatively low training time. Therefore, this value is considered as a reference (the results are presented in the corresponding section.). Indeed, experiments with datasets of 2'000 and 20'000 images yielded highly random and inconsistent results. For this reason, it is recommended that the dataset size be around 50'000 at least.

Additionally, the dataset has been tested with 2 million and 1 million images, and the results obtained were practically compatible with those observed using a dataset of 200'000 images. Therefore, if a dataset larger than 200'000 images is provided, a dataset of 200'000 images (100'000 for signal and 100'000 for background) will be randomly selected, and the models will be trained using this dataset. 

This ensures consistency and reliability in the results obtained from the training process.


# Versioni usate e pacchetti richiesti
This project was tested on macOS [Versione] Sonoma (M2 chip) with:
- ROOT version: ...
- Python version: ...
  - Pandas
  - Numpy
  - Torch
  - Tensorflow
  - Keras
  - Scikit-learn
  - Matplotlib
  - Xgboost
  - Uproot

# Referenza
Il progetto si è basato su uno dei tutorials di TMVA nella pagina web di ROOT, al seguente link: 'https://root.cern/doc/master/TMVA__CNN__Classification_8C.html'. Il codice originale è stato modificato (guardare paragrafo "Confronto rispetto alla referenza") con lo scopo di implementare quanto visto a lezione, e renderlo usabile per utenti sprovvisti di ROOT e/o Python ma in possesso di Docker.


# Confronto rispetto alla referenza


# Spiegazione Codice

## Gen_data.C
Il file "Gen_Data.C" si occupa di generare i dati per mezzo della seguente funzione "...". L'obiettivo è lasciare all'utente la possibilità di personalizzare le immagini e le distribuzioni dei dati sotto alcune condizioni, che sono le seguenti:
    - Le dimensioni possono essere solamente del tipo AxA quindi di dimensioni quadrate
    - I dati possono essere generati in accordo con le seguenti distribuzioni: Gaussiana, ...
    - Le immagini di segnali e background avranno le stesse dimensioni e le stesse distribuzioni
    - Se non specificato niente, il programma genera di default immagini 16x16 con dati distribuiti secondo una gaussiana.
Il codice lavora in questo modo: ....

## TMVA.C

## Keras.py

## Program_Start.py
This Python script is designed to perform a series of training and evaluation operations on different machine learning models, comparing them with each other. The process involves the use of convolutional neural networks (CNNs) implemented both with TensorFlow-Keras and PyTorch, a gradient boosting algorithm (BDT), and a densely connected neural network (DNN). The final comparison is done through the creation and visualization of ROC curves for each model.

### 1) DEPENDENCIES
The code relies on multiple Python libraries, including os, tensorflow, numpy, pandas, torch, matplotlib.pyplot, and time. Additionally, it incorporates functions defined in separate Python files, organized within specific directories: "Evaluation", "Models", and "DataPreparation". 
- Within the "Evaluation" directory, there are Python files such as "PredictionNN", "TrainTorch", "PredictionBDT", and "Table", each containing multiple functions. These functions handle tasks related to model prediction, PyTorch training, BDT prediction, and result tabulation, respectively.
- In the "Models" directory, there are individual Python files for different machine learning models, including "model_keras_noPCA", "model_keras_PCA", "model_DNN", "model_DNN_PCA", "model_Torch_noPCA", "model_Torch_PCA2, and "model_BDT".
- The "DataPreparation" directory contains Python files responsible for various data preprocessing tasks. These files include "Data_Preparation" (function: load_and_normalize_images, create_features_labels), "shuffle" (function: apply_shuffle), and "PCA" (functions: find_optimal_num_components, apply_pca), "Preparing_Dataset"(functions: Control_PCA) each containing functions for data loading and preprocessing, reshaping, shuffling, and Principal Component Analysis (PCA), respectively.

### 2) DATA LOADING & NORMALIZATION
- _Extraction Features_: The code extracts several variables from the file names: the number of events, stored as event_number, and the height and width of the images, stored as height and width, respectively. These variables serve as inputs for subsequent steps in the code.
- _File Path and Data Loading_: This functionality is implemented using the "load_and_normalize_images" function. The code takes two inputs: the folder path containing the images and the file name. It utilizes Uproot to open the specified file and loads the data for both signal and background images.
- _Data Normalization_: Implemented using the "load_and_normalize_images" function, this step calculates the maximum pixel values for both signal and background images. The images are then normalized by dividing each pixel by the calculated maximum value. This ensures that all pixel values are scaled between 0 and 1, making the data comparable and facilitating model training. To reduce computational burden, the code employs a batch approach, performing normalizations on batches of images instead of the entire dataset at once.

### 3) DATA PREPARATION
In this code section, we prepare the data for machine learning tasks by converting signal and background images into NumPy arrays and defining features and labels.
- _Data Conversion and Dimension Printing:_: Initially, we convert the signal and background images into NumPy arrays, which are stored as "signal_images_numpy" and "background_images_numpy," respectively. This conversion facilitates further processing of the image data. Additionally, we print the dimensions of the resulting NumPy arrays to look the shape of our data.
- _Features and Labels Definition_: Next, we utilize the "create_features_labels" function to define the features (X) and labels (y) for our machine learning model. The features (X) are generated by concatenating the signal and background data arrays. This concatenation merges the image data into a single array, which serves as the input for our model. Labels (y) are assigned to each data point, with signal data labeled as 0 and background data labeled as 1.

### 4) PREPARING DATASET: Principal Component Analysis (PCA)
The dataset preparation is handled by the "Control_PCA" function, which takes features (X), labels (y), image width, and height as inputs and performs the following tasks:
- If the product of the image width and height is greater than or equal to 144, indicating a large-dimensional dataset, PCA is applied. This is done by calling the "find_optimal_num_components" and "apply_pca" functions from "PCA.py". Additionally, the "apply_shuffle" function from "shuffle.py" is called to shuffle the data.
    - The "_find_optimal_num_components_" function calculates the optimal number of principal components required to explain a specified percentage (95%) of the total variance in the dataset. It achieves this by fitting a PCA model to the normalized data and computing the cumulative explained variance ratio to determine the number of components needed.
    - The "_apply_pca_" function applies PCA to the normalized data with the specified number of components (stored in the variable "num_components"). It returns the transformed data after dimensionality reduction.
    - The "_apply_shuffle_" function shuffles the data to introduce randomness, which helps prevent model overfitting and ensures robustness. It shuffles the data while maintaining the correspondence between features and labels.


The function returns the variables "i", "X_shuffled", "y_shuffled", and "n_principal_components". The variable "i" is necessary to distinguish between the case where PCA is applied and where it is not, while "n_principal_components" serves as input for 1D models.

### 4) MODEL CHOICE
The user is prompted to choose which model to start with. The logic behind this is as follows: all four models can be experimented with, but once a model is chosen, it cannot be retried in subsequent prompts. Additionally, pressing button number 5 will display a ROC curve comparing all the executed models. If only one model has been executed, only one ROC curve will be displayed. However, if all four models have been executed, then four ROC curves will be shown. The button 5 also acts as an "exit" button, allowing the user to terminate the program. The dataset is split into training and test sets, and the model is trained and evaluated for each model, except for PyTorch, where the data is first converted into tensors before training and evaluation. After that, the four models are defined within the if statement that manages the two scenarios: defining the model with PCA or without PCA. 

For the PyTorch-based model, we utilize the `trained_model` function to train the model, monitoring the time taken for training. Subsequently, we evaluate the model's performance on the test set using the `test_eval` function and generate predictions to compute metrics such as precision, F1 score, and accuracy. Finally, we visualize the ROC curve and training curves. Instead, for the Keras-based models (DNN and CNN), we train the model using Keras's `fit` method, monitoring the time taken for training. We then evaluate the model's performance on the test set and compute evaluation metrics, printing the ROC curve and training curves. Lastly, for the XGBoost-based model (BDT), we train the model using the `BDT_eval` function, evaluate its performance, and generate predictions to calculate evaluation metrics. Subsequently, we print the ROC curve and training curves. After each training and evaluation process of the models, parameters such as "f1", "accuracy", "precision", and "training time" are calculated and printed on a table. This table is updated each time with the results obtained from each model through the "table.py" module. Additionally, when button 5 is pressed, ROC curves of the trained models are generated and plotted.

#### 4.1) Train_Torch functions:
- _accuracy function_: This function computes the accuracy of the model's predictions compared to the target labels. It takes the model's predictions (outputs) and the target labels (targets) as input and returns a scalar value representing the accuracy.
- _trained_model_: This function manages the model training process. During each epoch, it performs both training and validation of the model. It utilizes a DataLoader to load the training and validation data in mini-batches. During training, it calculates the loss and accuracy of the model on the training data and updates the model's weights using the specified optimizer. During validation, it evaluates the model's performance on the validation data without updating the model's weights. The function returns the trained model and a dictionary containing the loss and accuracy values during training and validation.
- _test_eval_: This function evaluates the model's performance on the test data. It uses a DataLoader to load the test data in mini-batches. It calculates the loss and accuracy of the model on the test data and prints the results.
- _predict_: This function generates the model's predictions on the test data. It uses the trained model to make predictions on each mini-batch of test data and returns the predictions as a NumPy array.
- _torch_eval_: This function computes and returns various evaluation metrics of the model on the test data, including precision, F1 Score, and accuracy. It utilizes the predict function to obtain the model's predictions and compares the predictions with the target labels to calculate the metrics.
- _plot_training_curves_TORCH_: This function visualizes the loss and accuracy curves during the model training and validation process. It takes a dictionary containing the loss and accuracy values during training and validation as input and plots the corresponding curves using the Matplotlib library.

#### 4.2) Prediction_BDT functions:
- _BDT_eval_: This function evaluates the performance of the Boosted Decision Trees (BDT) model on the test data. It takes the trained model, the test data (X_test), and their corresponding labels (y_test) as input. The function utilizes the model to make predictions on the test data and calculates the accuracy, precision, and F1 score of the predictions. It returns the model's predictions, the rounded binary predictions, precision, F1 score, and accuracy.
- _print_ROC_BDT_: This function computes and displays the Receiver Operating Characteristic (ROC) curve and the confusion matrix of the Boosted Decision Trees (BDT) model on the test data. It takes the test labels (y_test), the model's continuous predictions (y_pred), and the rounded binary predictions (y_pred_binary) as input. The function computes the ROC curve using the continuous predictions and visualizes the confusion matrix. It returns the background rejection rate and the signal efficiency.
- _plot_training_curves_BDT_: This function visualizes the training and test curves of evaluation metrics for the Boosted Decision Trees (BDT) model. It takes the dictionary of evaluation metrics (evals_result) returned during the model training process and plots the log-loss curves for training and testing.


#### 4.3) PredictionNN functions:
- _eval_Neural_Networks_: This function evaluates the performance of the neural network on the test set. It takes the neural network model, the test data (X_test), and their corresponding labels (y_test) as input. The function utilizes the model's evaluate method to compute the loss and accuracy on the test data. It then makes predictions using the predict method and calculates the precision, F1 score, and accuracy of the predictions. The function returns the continuous predictions, rounded binary predictions, precision, F1 score, and accuracy.
- _print_ROC_: This function computes and displays the Receiver Operating Characteristic (ROC) curve and the confusion matrix of the neural network model on the test data. It takes the test labels (y_test), the model's continuous predictions (y_pred), and the rounded binary predictions (y_pred_classes) as input. The function calculates the ROC curve using the continuous predictions and visualizes the confusion matrix. It returns the background rejection rate and the signal efficiency.
- _plot_training_curves_KERAS_: This function visualizes the training and validation curves of the neural network. It takes the history object returned during the neural network training process and plots the accuracy and loss curves for training and validation.





"UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)" __pendente__





## Script.sh
- _Project Cleanup_: Removes directories from previous runs, including "ROOT/images", "Python_code/images", "Python_code/plot_results", and all "pycache" folders within "Python_code" and its subdirectories. This task is handled by the "generate_delete_command" function
- _ROOT and Python Verification_: Checks for the presence of ROOT and Python on the computer and ensures their compatibility with the tested code version. If Python and/or ROOT are not present, or if their versions are incompatible, the dockerfile is executed. Additionally, it verifies the presence and version of pip for potential Python library installation. This code is executed by the "check_python", "check_root_version", and "check_pip" functions.
- _Library Compatibility Check and Update_: Asks the user if they want to verify the presence and compatibility of necessary libraries. If the user chooses not to verify, a message displaying the versions used is printed to the terminal. If the user opts for verification, missing libraries are installed if desired, otherwise the user is prompted to use the dockerfile. If libraries are present but outdated, they are updated to the tested version. If they are more recent than the tested version, they are adjusted. Both actions are performed only if the user desires; otherwise, the code continues its execution, although consequences may be unknown. This code is executed by the "update_python_libraries" function.
- _Dataset Generation_:Asks the user if they want to generate their dataset or use the default one. If chosen to generate, the dataset is created using "Generation.C", allowing the user to specify the number, height, and width of images. If chosen not to generate, the dataset is downloaded using the wget method. In both cases, a "images" folder is created where the dataset is placed. This part of the code is managed by the "Gen_files" function. (__fare il wget__)
- _Copying Dataset_:The "images" folder containing the dataset is copied and moved to the "CNN_python" and "TMVA_ML" directories. This code is executed by "move_images_folders" function.
-  _File Presence Verification_: Verifies the presence of the file in various subdirectories.This code is executed by "check_root_file" function. 
-  _Training Environment Selection_: Asks the user to choose whether to start training with ROOT or Python. Then, the user can decide if they want to experiment with the other file as well. This is managed in the main script through keyboard commands.

# How to run

Once you're ready, clone the repository:

$ git clone https://github.com/gaiafabbri/S-C.git 

Navigate to the project directory:

$ cd your/path/to/S-C
## Using script file

Obtain permissions to execute the script:

$ chmod +x Script.sh

Execute the bash script:

$ ./Script.sh

## Run individual files

Alternatively, you can run individual files. First, ensure you have a dataset. You can either download it using:

$ wget...

Or generate the dataset using the provided macro:

$ .L [nome macro]

$ nome funzione (n, nh, nw)

Once you have the dataset, move it to the "Python_code" and "TMVA" directories. Now you can execute the following commands:

Inside "TMVA":

$ .L [Nome file]

$ Nome funzione

Inside "Python_code":

$ python3 Program_start.py

## Only dockerfile

If you prefer to run the dockerfile directly, follow these instructions:

$


