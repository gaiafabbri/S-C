# S-C


# Project description
The following project aims to classify signal and background images using machine learning techniques. Ci sono due file principali per l'addestramento sul dataset (vedi Folders organisation): uno basato su TMVA e l'altro sui framework definiti da Python come Tensorflow-Keras, Torch, BDT. 

in particular, a comparison is made between TMVA (Toolkit for Multi-Variate Analysis) packages and Python libraries. The former includes CNN, DNN and Boosted Decision Tree (BDT); the latter implements the same algorithms: CNN is permormed with both keras-tensorflow and pytorch, BDT uses xgboost classifier, while DNN classification is based on keras-tensorflow methods. The models are evaluated in terms of precision, accuracy, f1-score and roc curve



# Folders organisation 

The project structure includes:

- A folder named "ROOT_Gen" that holds a file named "Generation.C". This file is responsible for generating the dataset of images which are saved into "images" subfolder to be used. For more details on dataset generation, refer to the dataset section.
- Another folder named "CNN_python", which encompasses several subfolders and generates an additional folder post-code execution: plot_results" which aids in visualizing training parameters alongside the Receiver Operating Characteristic (ROC) curve. This curve illustrates signal efficiency versus background rejection for each model. The "plot_results" folder also contains a comparative visualization of ROC curves and a table showcasing performance metrics such as f1 score, accuracy, precision, and training time across different models.
    - There's "Program_Start.py" which serves as the main execution script for machine learning methods. These methods include Convolutional Neural Networks (CNN) implemented with both Torch and TensorFlow-Keras, as well as Boosted Decision Trees (BDT) and Deep Neural Networks (DNN) with Keras. The script acts as an interface with the user, orchestrating functions imported from Python scripts in the subfolders to create a unified training file according to user-selected models. Detailed explanations are provided within the code comments.
    - The "DataPreparation" subfolder houses scripts for data preparation, including:
        - "Data_Preparation.py" for loading data into arrays and subsequent normalization.
        - "reshaping.py" for reshaping data, particularly useful when training involves image data (2D).
        - "PCA.py" for data preparation when training involves numpy arrays (1D).
        - "shuffle.py" for shuffling dataset entries.
    - The "Models" subfolder contains various model implementations, each available in both 1D (after Principal Component Analysis, PCA) and 2D versions (without PCA). Notably, the BDT implementation is an exception. Files within this subfolder include:
         - "model_BDT.py"
         - "model_DNN_PCA.py"
         - "model_DNN.py"
         - "model_keras_noPCA.py"
         - "model_keras_PCA.py"
         - "model_Torch_noPCA.py"
         - "model_Torch_PCA.py"
      - Lastly, the "Evaluation" subfolder hosts files for assessing model performance, including:
        - "PredictionNN.py", designed for both CNNs implemented with Keras-TensorFlow.
        - "PredictionBDT.py"
        - "TrainTorch.py"
        - "Table.py", responsible for tabulating results from each model (f1 score, accuracy, precision), highlighting the best-performing model. Additionally, it includes a column displaying the training time for each investigated model.
- Inside the project, there's also a folder named "TMVA_ML". This folder contains a file named "TMVA.C". When executed with the provided bash file (see below), it will generate an additional folder named "images". This folder is generated using "Generation.C" and contains the dataset created. The methods implemented within "TMVA.C" include a Convolutional Neural Network (CNN), Deep Neural Network (DNN), and Boosted Decision Tree (BDT). This file originates from an example provided in the ROOT tutorials, accessible via the following link: 'https://root.cern/doc/master/TMVA__CNN__Classification_8C.html'
- An executable bash file called 'Script.sh' that has the task of:
    - __Project Cleanup__: Removes directories from previous runs, including "ROOT/images", "Python_code/images", "Python_code/plot_results", and all "pycache" folders within "Python_code" and its subdirectories. This task is handled by the "generate_delete_command" function
    - __ROOT and Python Verification__: Checks for the presence of ROOT and Python on the computer and ensures their compatibility with the tested code version. If Python and/or ROOT are not present, or if their versions are incompatible, the dockerfile is executed. Additionally, it verifies the presence and version of pip for potential Python library installation. This code is executed by the "check_python", "check_root_version", and "check_pip" functions.
    - __Library Compatibility Check and Update__: Asks the user if they want to verify the presence and compatibility of necessary libraries. If the user chooses not to verify, a message displaying the versions used is printed to the terminal. If the user opts for verification, missing libraries are installed if desired, otherwise the user is prompted to use the dockerfile. If libraries are present but outdated, they are updated to the tested version. If they are more recent than the tested version, they are adjusted. Both actions are performed only if the user desires; otherwise, the code continues its execution, although consequences may be unknown. This code is executed by the "update_python_libraries" function.
    - __Dataset Generation__:Asks the user if they want to generate their dataset or use the default one. If chosen to generate, the dataset is created using "Generation.C", allowing the user to specify the number, height, and width of images. If chosen not to generate, the dataset is downloaded using the wget method. In both cases, a "images" folder is created where the dataset is placed. This part of the code is managed by the "Gen_files" function. (__fare il wget__)
    - __Copying Dataset__:The "images" folder containing the dataset is copied and moved to the "CNN_python" and "TMVA_ML" directories. (__This part of the code is pending implementation__).
    -  __File Presence Verification__: Verifies the presence of the file in various subdirectories. (__This part of the code is pending implementation__).
    -  __Training Environment Selection__: Asks the user to choose whether to start training with ROOT or Python. Then, the user can decide if they want to experiment with the other file as well. This is managed in the main script through keyboard commands.

- Il "dockerfile" che è stato costruito nel caso in cui l'utente non abbia ROOT e/o Python. Esso crea un ambiente per poter usare sia ROOT, sia Python indipendentemente dalla presenza di uno o dell'altro sul computer. In termini pratici sostituirà l'eseguibile "run.sh" replicandone i compiti e la gestione del progetto.


# Data Generation
Il dataset viene generato utilizzando ROOT (vedi Folders organisation) e chiedendo all'utente se usare un file generato di default 16x16 con 100'000 eventi per segnale e per rumore, oppure se generarlo lui, e in quel caso l'utente avrà il libero arbitrio di generare il dataset delle dimensioni che vorrà e con immagini del formato che desidera, anche non quadrata. __COMMENTARE__



The code has been tested on the default dataset but adapted to cover a general scenario. Regarding the analysis conducted in Python, two preprocessing strategies are explored before training: with PCA or without PCA. The idea is that models could be either 1D or 2D, specifically in reference to Convolutional Neural Networks (CNN) using Keras-TensorFlow and Torch, as well as Deep Neural Networks (DNN). This depends on the dimensions of the dataset because, obviously, if we have 2x2 images, PCA will not be applied.

A threshold is established around the image dimensions, such as 10x10 or any combination AxB where both A and B are less than 10. In cases where A or B exceeds 10, only the sum of A and B is considered, which must not exceed 20. If the images are square and exceeds 11x11 or if A+B exceeds 20, PCA will be applied to enable training with 1D models. It is recommended that the dataset size be around 50,000 at least. Values below 10,000 may yield rather random and certainly insignificant results.





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

## PyTorch.py


# How to run

Once you're ready, clone the repository:

$ git clone https://github.com/gaiafabbri/S-C.git 

Navigate to the project directory:

$ cd your/path/to/S-C

Obtain permissions to execute the script:

$ chmod +x Script.sh

Execute the bash script:

$ ./Script.sh


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

If you prefer to run the dockerfile directly, follow these instructions:

$


