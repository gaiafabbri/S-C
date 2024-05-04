# S-C Project description
The project compares machine learning techniques for classifying signal and background images, utilizing two main approaches: one based on the TMVA (Toolkit for Multi-Variate Analysis) packages and the other on Python frameworks such as TensorFlow-Keras, PyTorch, and XGBoost for BDT classification. In particular, the comparison includes Convolutional Neural Network (CNN), Deep Neural Network (DNN), and Boosted Decision Trees (BDT) algorithms. For TMVA, these algorithms are employed directly, while in Python, CNN is implemented using both Keras-TensorFlow and PyTorch, BDT uses XGBoost classifier, and DNN classification is based on Keras-TensorFlow methods. The models are evaluated based on precision, accuracy, F1-score, and ROC curve.

The results are obtained using a dataset of 16x16 images, containing 100,000 images each for signal and background (totaling 200,000 images). This dataset is generated with parameters such as: 100,000 events (n), image height and width of 16 bins each (nh = nw = 16), and each bin is filled with 10,000 events (nRndmEvts). The signal and background distributions have a 5% difference in standard deviation (delta_sigma), and Gaussian noise of 5 is added to each bin of the signal (pixelNoise). Within each bin, data are generated using Gaussian distributions centered on the values obtained from the fits of two functions, with additional Gaussian noise.

In Python, the dataset undergoes normalization of pixel data to a range of 0 to 1, label creation for signal and background (assigned numbers 0 and 1 respectively), and Principal Component Analysis (PCA) for dimensionality reduction. This preprocessing is applied to neural network models, while BDT utilizes normalized data with labels for signal and background without PCA. The results of the comparison among various machine learning techniques are presented below. 
![Comparison_among_models_100000_16x16](https://github.com/gaiafabbri/S-C/assets/133896928/f07ba211-bb44-4383-aafb-3172c9b1635c)


![arturo](https://github.com/gaiafabbri/S-C/assets/133896928/fe4193e5-8c68-44ee-ab28-5d7b722e5f24)

The majopr aim of this project is to perform a comparison between the TMVA tutorials "CNN_Classification.C" available on the ROOT website at the link: 'https://root.cern/doc/master/TMVA__CNN__Classification_8C.html' and the Python script; the original codes contains also a PyKeras and PyTorch classification. This part was extracted and re-written in order to apply what learned during the lessons and to improve my knoledge about machine learning tecnique. The same method are implemented in the same script: the TMVA coe was only modify and taken as a reference to build the python analysis. The implementation of the method is surely different passing from root to python environment, and this affect also performances (see later the paragraph about the results).

For TMVA, the dataset is normalized using NormMode=NumEvents (average weight of 1 per event, independently for signal and background, eand the models are trained  and tested accordingly to the configuration options. Results are visualized to compare the performance of different algorithms. The results of the comparison among various machine learning techniques are presented below. 

##METTERE IMMAGINE QUI!!!!####

## Principal Component Analysis description

Principal Component Analysis is used for both the TMVA and the Python analysis: it consists of a data transformation aimed at reducing the components of each image. To be more precise, each image is composed of 256 components, but not all of them are relevant for classification; the PCA transformation reduces the number of components in each image by finding the directions along which the features show the greatest variations. It helps to reduce the complexity of the data and at the same time improves the performance of the model; it is usually applied to high resolution images and it is also useful to avoid correlations between variables: the CNN classification seems to be more effective when the variables are not correlated. In the TMVA script, the PCA is entered directly by setting "Transformations=P" in the Factory object; in the Python analysis, the PCA is implemented in both the Keras and PyTorch models for the CNN, which requires more complex model definitions; moreover, these two models require a complex shape of the data, two-dimensional for the former and tensors for the latter. PCA transforms this shape into a one-dimensional array, which is easier to manage. 

CNN is a classification tecnhique very suitable for images or grid data, but it is specific and the model is difficult to be build. On the contrary, DNN and BDT are simpler but more robust models with a one-dimensional input: the PCA is not applied to BDT since it is shows good performances instead and so it is chosen as a reference; DNN and CNN intead shows a steep increase in performance after PCA implementation, and they are alsd faster.
The BDT and DNN models are more generic in their application, and their advantages are flexibility and ability to recognise patterns for the former, easy interpretation and robustness for the latter. The DNN are made up of different layers of artificial neurons, while the BDT is based on a decision tree, trying to correct errors made by the previous tree, continuing the training in this way.

For further details and explanations regarding the code implementation, please refer to the specific section.


# Comments on results obtained
For the TMVA analysis, the most performing model seems to be the CNN, followed immediately by the DNN; the BDT semms a little worst in classification. The ROC-integer for each model is respectively 0.941, 0.937 and 0.821, with the CNN which shows also the better signal efficiency (0.955, 0,950 and 0.781 for the 30% of background, respectively). CNN and DNN show very similar behaviour, while the BDT model is a little more discriminative, but still performs an efficient classification. However, the BDT shows a fast training, with 32 s for training and 0.65 s for evaluation; the DNN is also faster, with 17 s for training and 1.2 s for evaluation. The CNN is significantly slower, with 170 s for training and 8 s for evaluation.
As far as the Python analysis is concerned, the results between the different models are very similar and all the models seem to perform quite well in the classification. The Torch CNN is the one with the higher accuracy and F1 score, while the DNN has the better accuracy; however, the values for the other model are not far from the optimum and in general they are about 86% for all the models. Again, the CNN with Keras has the slowest training time of 73 s, but the other models take 11 s, 5 s and 18 s for Torch CNN, BDT and DNN respectively. Confusion matrices were also computed to evaluate the performance of the classifiers: on the test dataset of 40,000 events, all models show about 17,000 true positives and true negatives, for a total of 34,000 correctly identified events; only the BDT performs slightly worse, with about 16,000 true positives and 16,000 true negatives. The ROC curves obtained from the Python analysis are quite consistent with the TMVA curves, as the CNN (both with Keras and PyTorch) and DNN curves almost overlap, while the BDT curve is slightly lower.
In conclusion, CNNs are generally preferred for image classification due to
-Their spatial invariance, which means that they are able to identify patterns even if they do not have the same position within the image.
-The use of more layers makes the CNN suitable for learning complex patterns, since the first layers learn the simpler patterns, which are then used to learn more complex representations.
Keras provides a user-friendly environment with predefined layers, making it easy for less experienced users to construct high-performance networks; PyTorch is more flexible, as the computational design of the network is defined dynamically and modified during the program run. However, it seems to be more complicated to implement. BDT and DNN are more flexible and generic models that seem to perform quite well in classifying background and signal images anyway: they are a suitable choice for image classification if the dataset is not too complicated and they provide easily interpretable results with reduced training time. In general, CNN are more complex models to train, but they can lead to very high performances; however, for our dataset, CNN leads to a slower training time without a significant improvement in the performances, especially with respect to the DNN model.

# Folders organisation 

The project structure includes:

- A folder named "__ROOT_Gen__" that holds a file named "Generation.C". This file is responsible for generating the dataset of images which are saved into "images" subfolder to be used. For more details on dataset generation, refer to the dataset section.
- Another folder named "__Python_code__", which encompasses several subfolders and generates an additional folder post-code execution: "plot_results" which aids in visualizing training parameters alongside the Receiver Operating Characteristic (ROC) curve. This curve illustrates signal efficiency versus background rejection for each model. The "plot_results" folder also contains a comparative visualization of ROC curves and a table showcasing performance metrics such as f1 score, accuracy, precision, and training time across different models.
    - There's "_Program_Start.py_" which serves as the main execution script for machine learning methods. These methods include CNN implemented with both Torch and TensorFlow-Keras, as well as BDT and DNN with Keras. The script acts as an interface with the user, orchestrating functions imported from Python scripts in the subfolders to create a unified training file according to user-selected models. Detailed explanations are provided within the code comments.
    - The "_DataPreparation_" subfolder houses scripts for data preparation, including:
        - "Data_Preparation.py" for loading data into arrays and subsequent normalization.
        - "PCA.py" for data preparation when training involves numpy arrays (1D).
        - "Preparing_Dataset.py" is designed to facilitate the preparation of datasets for the machine learning techniques.
    - The "_Models_" subfolder contains various model implementations, each available in 1D (after PCA, except BDT). Files within this subfolder include:
         - "model_BDT.py"
         - "model_DNN_PCA.py"
         - "model_keras_PCA.py"
         - "model_Torch_PCA.py"
      - Lastly, the "_Evaluation_" subfolder hosts files for assessing model performance, including:
        - "PredictionNN.py", designed for both NNs implemented with Keras-TensorFlow.
        - "PredictionBDT.py"
        - "TrainTorch.py"
        - "Table.py", responsible for tabulating results from each model (f1 score, accuracy, precision), highlighting the best-performing model. Additionally, it includes a column displaying the training time for each investigated model.
- Inside the project, there's also a folder named "__TMVA_ML__". This folder contains a file named "TMVA.C". When executed with the provided bash file (see below), it will generate an additional folder named "images". This folder is generated using "Generation.C" and contains the dataset created. The methods implemented within "TMVA.C" include a Convolutional Neural Network (CNN), Deep Neural Network (DNN), and Boosted Decision Tree (BDT). This file originates from an example provided in the ROOT tutorials, accessible via the following link: 'https://root.cern/doc/master/TMVA__CNN__Classification_8C.html'

- An executable bash file called '__Script.sh__' which serves as a tool for project setup and management. It automates various tasks such as project cleanup, verification of ROOT and Python installations, library compatibility checks, and dataset generation. It allows users to select their preferred training environment (ROOT or Python).

- 
- Il "__dockerfile__" che è stato costruito nel caso in cui l'utente non abbia ROOT e/o Python. Esso crea un ambiente per poter usare sia ROOT, sia Python indipendentemente dalla presenza di uno o dell'altro sul computer. In termini pratici sostituirà l'eseguibile "run.sh" replicandone i compiti e la gestione del progetto.




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

# Script description
This section will examine the codes in detail, elucidating their operational characteristics. The aim is to provide a comprehensive analysis of their functionalities, elucidating each step to comprehensively understand what they do and how they accomplish it. 

## Generation_Images.C
The dataset generation process utilizes the Generation.C module within the ROOT folder through the function "Generation". It prompts the user to choose between using a default file generated with dimensions 16x16 and 100,000 events for signal and background, or generating a custom dataset. In the latter case, the user has the freedom to define the dataset dimensions and image format, even non-square ones, with certain constraints. It is necessary to specify that the code was tested woth the default configuration, so we expect differen performances if the dataset is changed.

The "Generation" function requires three parameters: the number of images to generate ("n"), the height ("nh"), and the width ("nw") of the images. It defines a two-dimensional Gaussian distribution for both the signal and background using the ROOT TF2 class. The parameter "nRndmEvts" determines the number of events used to fill each image, representing the number of random events sampled from the signal and background distributions to create a single image example. A higher value of "nRndmEvts" can lead to more defined and realistic images but requires more computational resources to generate the data. Therefore, it is set to 10,000 (non-modifiable). The parameter "delta_sigma" represents the percentage difference in standard deviation (sigma) between the signal and background distributions. A higher value increases the difference between the widths of the background and signal distributions, making them easier to be distinguished. It is also set to 5% and is non-modifiable. Random noise is added to the signal, linked to the variable "pixelNoise", representing the level of noise added to each pixel of the generated image. It measures the dispersion or random variation of pixel values.

The variables "sX1", "sY1", "sX2", "sY2" represent the sigma values along the x and y axes for the first and second Gaussian distributions, respectively, where "sX2" is increased by 5% compared to "sX1", and "sY2" is decreased by 5% compared to "sY1". Two TH2D histograms are created using ROOT to store the data, along with two TF2 functions representing the Gaussian distributions. Two TTree trees are then created: one for signal data (sig_tree) and one for background data (bkg_tree). Branches of the trees ("vars") are defined to contain the image data, using a pointer to a float vector to store the data.


 The two functions, "f1.SetParameters(1, 5, sX1, 5, sY1)" and "f2.SetParameters(1, 5, sX2, 5, sY2)" are used to set the distribution of data within the images and are defined with the following parameters, in order:
- Maximum height of the Gaussian
- Mean position along the x-axis
- Standard deviation along the x-axis
- Mean position along the y-axis
- Standard deviation along the y-axis

Inside the firstloop, the instructions "h1.FillRandom("f1", nRndmEvts)" and "h2.FillRandom("f2", nRndmEvts)" fill the histograms h1 and h2 with randomly generated data using the functions f1 and f2, respectively. The FillRandom() function takes the name of a function (in this case, f1 and f2) as an argument and generates random values for the histogram variables distributed according to the specified function's distribution.

A second loop iterates through all cells (or bins) of the image, where "nh" represents the number of rows and "nw" represents the number of columns of the image. This loop traverses through each row and column of the image, allowing access and manipulation of each individual bin of the two-dimensional histogram. An index "m" is calculated for each bin, representing the position of the bin in the two-dimensional array x1 and x2. This is done by multiplying the row index by the number of columns and adding the column index. This index "m" is used to access the vectors x1 and x2, which contain the image data. For each bin of the image, random Gaussian noise is added using the function "gRandom->Gaus(0, pixelNoise)", which generates a random number distributed according to a Gaussian distribution with mean 0 and standard deviation "pixelNoise". This noise is added to the value of the bin obtained from the respective histograms h1 and h2, resulting in image data with added noise.

## TMVA_Classification.C ---vedere forma finale-----

### Preparing the environment and the dataset
This code is an application which uses the TMVA (Toolkit for Multivariate Analysis) environement to classify signal and background images implementing a CNN, a DNN and a BDT TMVA methods; the analysis is performed througout the function TMVA_Classification()', which takes as input the number of 'nevts' and the 'opt' vector containign the possible methods used for classification. 

The size of the input images and the number of events is then extracted from the file name; the analysis is performed in multithread option, with n_threads=4: if the latter is not available the DNN method cannot be run. A file root containing the output is created and the analysis starts with the creation of a factory object, containing the configuration options: in particular, the data undergo PCA transformation ("Transformations=P"); other options include the verbose flag the verbose flag, the flag for coloured screen output (Colour=True) and the flag for output visualisation ("!Silent") 
The training dataset is loaded using the DataLoader object, which is then used to specify the input variables: the AddVariable method takes as arguments the name as a string and the number of elements in the array, and the input variable is an array of 256 variables; there are no spectator variables. Signal and background events are extracted from two different trees and the same weight is given to both classes; individual weights can be assigned to two classes. The number of events selected for training is the 80% of the total entries of the trees, both for signal and background; at this point the dataset is ready for training.

### Choice and Training of TMVA methods 
The following segment of the code is dedicated to configuring TMVA methods for training classification models (BDT, CNN, and DNN with TMVA). In TMVA, all methods are trained, tested and evaluate together as follows:
- "factory.TrainAllMethods()": This method trains all the specified methods (in this case, BDT, DNN, and CNN) using the previously prepared training data.
- "factory.TestAllMethods()": It tests all the trained methods using the pre-prepared test data.
- "factory.EvaluateAllMethods()": This step evaluates the performance of all the trained methods using appropriate metrics.
- "auto c1 = factory.GetROCCurve(&loader)": It obtains the ROC curve for all the trained methods using the input data provided by the DataLoader named loader.

#### 1) TMVA_BDT
If "useTMVABDT" is true, the boosted decision tree (BDT) method is booked using "factory.BookMethod". Various options are specified for the BDT:
- "NTrees=200": Specifies the number of decision trees (or estimators) to use in the BDT. Here, 200 trees are used.
- "MinNodeSize=2.5%": Sets the minimum node size, controlling the minimum number of events required in a node for further splitting. Here, it's set to 2.5% of the total number of events.
- "MaxDepth=2": Specifies the maximum depth of the decision tree. Limiting the tree depth can help prevent overfitting.
- "BoostType=AdaBoost": Specifies the boosting type to use. Here, AdaBoost is used.
- "AdaBoostBeta=0.5": Sets the beta parameter of AdaBoost, controlling the importance of the weight of previous errors versus current ones.
- "UseBaggedBoost": Enables the use of bagging along with AdaBoost. Bagging enhances model stability and accuracy by reducing variance.
- "BaggedSampleFraction=0.5": Specifies the fraction of events to use for training each tree within bagging. Here, it's set to 50%.
- "SeparationType=GiniIndex": Specifies the separation criterion used for splitting nodes during tree construction. Here, Gini index is used.
- "nCuts=20": Specifies the maximum number of cut points to be tested for each variable predicate. A higher number may lead to higher precision but may also increase training time.

  
#### 2) TMVA_DNN
This code block books the deep neural network (DNN) method via TMVA. Here's an explanation of the provided options:
- "Layout": Defines the neural network architecture. In this case, the network has four dense layers with 64 neurons each, using ReLU activation and batch normalization. The last layer is a single neuron with linear activation, typically used for classification problems.
- "TrainingStrategy": Specifies the training strategies for the DNN. It includes parameters like learning rate, momentum, repetitions, convergence steps, batch size, etc...
- "Architecture": Specifies the neural network architecture (CPU or GPU) depending on availability. If TMVA is compiled with GPU support, the GPU architecture will be used; otherwise, the CPU architecture will be used.

A more detailed description is given below: the dense layers are formed by neurons completely connected to the output of the following layers; the number of neurons is chosen to be 64, since the default value of 100 degrades the performance, resulting in models too complex. A ReLU activation layer and a BNORM normalisation layer are introduced to allow the network to learn complex non-linear relations, the former and a more stable and faster training, the latter. Finally, the last linear with 1 neuron is usually chosen for a binary classification problem since it only has to report the probability of belonging to the signal or background class. 

#### 3) TMVA_CNN
This code block books the convolutional neural network (CNN) method via TMVA. Here's an explanation of the provided options:
- "Input Layout": Specifies the input layout of the network. Here, the input image has a single channel (grayscale) with a height and width of 16 pixels each
- "Batch Layout": Specifies the input batch layout. Here, the batch size is 100, the number of channels is 1, and the image size is 16x16.
- "Layout": Defines the CNN architecture, including convolutional layers, MaxPooling layer, Reshape layer, and dense layers.
- "TrainingStrategy": Specifies the training strategies for the CNN, similar to those for the DNN.
- "Architecture": Specifies the neural network architecture (CPU or GPU) depending on availability, similar to the DNN setup.

The network is composed of two convolutional layers, each followed by a ReLU activation layer and a BNORM normalisation layer, similar to the DNN; two dense layers are added, each with 64 neurons: again, better performance is obtained by reducing the number of neurons. The MAXPOOL layer reduces the spatial dimension of the representation, keeping only the most important feature, while the RESHAPE layer transforms the data into a flat array that is passed to the dense layer. The final layer is again a single neuron linear layer, which is well suited to binary classification problems.

## Program_Start.py
This Python script is designed to perform a series of training and evaluation operations on different machine learning models, comparing them with each other. The process involves the use of convolutional neural networks (CNNs) implemented both with TensorFlow-Keras and PyTorch, a gradient boosting algorithm (BDT), and a deep connected neural network (DNN). The final comparison is done through the creation and visualization of ROC curves for each model.

### 1) DEPENDENCIES
The code relies on multiple Python libraries, including os, tensorflow, numpy, pandas, torch, matplotlib.pyplot, and time. Additionally, it incorporates functions defined in separate Python files, organized within specific directories: "Evaluation", "Models", and "DataPreparation". 
- Within the "Evaluation" directory, there are Python files such as "PredictionNN", "TrainTorch", "PredictionBDT", and "Table", each containing multiple functions. These functions handle tasks related to model prediction, PyTorch training, BDT prediction, and result tabulation, respectively.
- In the "Models" directory, there are individual Python files for different machine learning models, including "model_keras_PCA", "model_DNN_PCA", "model_Torch_PCA", and "model_BDT".
- The "DataPreparation" directory contains Python files responsible for various data preprocessing tasks. These files include "Data_Preparation" (function: load_and_normalize_images, create_features_labels and apply_shuffle), "PCA" (functions: find_optimal_num_components, apply_pca), "Preparing_Dataset"(functions: Control_PCA). Each module contains functions for data loading and preprocessing, reshaping, shuffling, calculating Principal Component Analysis (PCA), and applying PCA directly to a dataset, depending on the ML algorithm in question.
- 
### 2) DATA LOADING & NORMALIZATION
- _Extraction Features_: The code extracts several variables from the file names: the number of events, stored as event_number, and the height and width of the images, stored as height and width, respectively. These variables serve as inputs for subsequent steps in the code.
- _File Path and Data Loading_: This functionality is implemented using the "load_and_normalize_images" function. The code takes two inputs: the folder path containing the images and the file name. It utilizes Uproot to open the specified file and loads the data for both signal and background images.
- _Data Normalization_: Implemented using the "load_and_normalize_images" function, this step calculates the maximum pixel values for both signal and background images. The images are then normalized by dividing each pixel by the calculated maximum value. This ensures that all pixel values are scaled between 0 and 1, making the data comparable and facilitating model training.
  

### 3) DATA PREPARATION
In this code section, we prepare the data for machine learning tasks by converting signal and background images into NumPy arrays and defining features and labels.
- _Data Conversion and Dimension Printing:_: Initially, we convert the signal and background images into NumPy arrays, which are stored as "signal_images_numpy" and "background_images_numpy," respectively. This conversion facilitates further processing of the image data. Additionally, we print the dimensions of the resulting NumPy arrays to look the shape of our data.
- _Features and Labels Definition_: Next, we utilize the "create_features_labels" function to define the features (X) and labels (y) for our machine learning model. The features (X) are generated by concatenating the signal and background data arrays. This concatenation merges the image data into a single array, which serves as the input for our model. Labels (y) are assigned to each data point, with signal data labeled as 0 and background data labeled as 1.

### 4) PREPARING DATASET: Principal Component Analysis (PCA)
The dataset preparation is handled by the "Control_PCA" function, which takes features (X), labels (y), image width, and height as inputs and performs the following tasks:
- In the event that the selected model is one of the DNN, CNN with Torch or TensorFlow-Keras, the code will perform PCA on the dataset. This is achieved by invoking the "find_optimal_num_components" and "apply_pca" functions from "PCA.py" and additionally calling the "apply_shuffle" function from "Data_Preparation.py" to shuffle the data.
    - The "_find_optimal_num_components_" function calculates the optimal number of principal components required to explain a specified percentage (95%) of the total variance in the dataset. It achieves this by fitting a PCA model to the normalized data and computing the cumulative explained variance ratio to determine the number of components needed.
    - The "_apply_pca_" function applies PCA to the normalized data with the specified number of components (stored in the variable "num_components"). It returns the transformed data after dimensionality reduction.
    - The "_apply_shuffle_" function shuffles the data to introduce randomness, which helps prevent model overfitting and ensures robustness. It shuffles the data while maintaining the correspondence between features and labels.
- If the model is instead BDT, the X and y datasets are left unaltered and recalled as X_new.
It is worth emphasising that the BDT model does not require data shuffling, since each decision tree is built independently of the order of the data and there is no sequential dependency as in neural networks; this means that the order of the data does not affect the training. On the other hand, in the DNN and CNN methods, the order of the data affects the learning; moreover, the shuffle prevents the model from learning patterns that are only present in a given order of the data.

The function returns the variables "X_new", "y_new", and "n_principal_components". The code is initiated at the outset of each model.

### 4) MODEL CHOICE
The user is prompted to choose which model to start with. The logic behind this is as follows: all four models can be experimented with, but once a model is chosen, it cannot be retried in subsequent prompts. Additionally, pressing button number 5 will display a ROC curve comparing all the executed models. If only one model has been executed, only one ROC curve will be displayed. However, if all four models have been executed, then four ROC curves will be shown. The button 5 also acts as an "exit" button, allowing the user to terminate the program. The dataset is split into training and test sets, and the model is trained and evaluated for each model, except for PyTorch, where the data is first converted into tensors before training and evaluation. After that, the four models are defined. 

For the PyTorch-based model, we utilize the `trained_model` function to train the model, monitoring the time taken for training. Subsequently, we evaluate the model's performance on the test set using the `test_eval` function and generate predictions to compute metrics such as precision, F1 score, and accuracy. Finally, we visualize the ROC curve and training curves. Instead, for the Keras-based models (DNN and CNN), we train the model using Keras's `fit` method, monitoring the time taken for training. We then evaluate the model's performance on the test set and compute evaluation metrics, printing the ROC curve and training curves. Lastly, for the XGBoost-based model (BDT), we define and train the model using the 'BDT_model' function, while the `BDT_eval` function evaluates its performance, and generates predictions to calculate evaluation metrics. Subsequently, we print the ROC curve and training curves. After each training and evaluation process of the models, parameters such as "f1", "accuracy", "precision", and "training time" are calculated and printed on a table. This table is updated each time with the results obtained from each model through the "table.py" module. Additionally, when button 5 is pressed, ROC curves of the trained models are generated and plotted.

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

#### 4.4) Table function:
Inside the Table file, the save_results_table function is designed to save the evaluation metrics results of the models (f1, accuracy, precision) and the training time into a Pandas DataFrame, and then draw a table using Matplotlib. Here are the parameters:
- "model_results":  A dictionary containing the evaluation metrics results for each model. The keys of the dictionary are the model names, and the values are tuples containing accuracy, precision, f1 score, and training time.
- "results_df": The Pandas DataFrame containing the models' evaluation results. If it's None (i.e., when the code is run for the first time), a new DataFrame will be created.

The function iterates through the model results in the "model_results" dictionary. For each model, it extracts accuracy, precision, f1 score, and training time from the respective tuple and adds them as a new row to the "results_df" DataFrame. Finally, it draws a table containing the results.








## Script.sh ---vedere forma finale-----
- _Project Cleanup_: Removes directories from previous runs, including "ROOT/images", "Python_code/images", "Python_code/plot_results", and all "pycache" folders within "Python_code" and its subdirectories. This task is handled by the "generate_delete_command" function
- _ROOT and Python Verification_: Checks for the presence of ROOT and Python on the computer and ensures their compatibility with the tested code version. If Python and/or ROOT are not present, or if their versions are incompatible, the dockerfile is executed. Additionally, it verifies the presence and version of pip for potential Python library installation. This code is executed by the "check_python", "check_root_version", and "check_pip" functions.
- _Library Compatibility Check and Update_: Asks the user if they want to verify the presence and compatibility of necessary libraries. If the user chooses not to verify, a message displaying the versions used is printed to the terminal. If the user opts for verification, missing libraries are installed if desired, otherwise the user is prompted to use the dockerfile. If libraries are present but outdated, they are updated to the tested version. If they are more recent than the tested version, they are adjusted. Both actions are performed only if the user desires; otherwise, the code continues its execution, although consequences may be unknown. This code is executed by the "update_python_libraries" function.
- _Dataset Generation_:Asks the user if they want to generate their dataset or use the default one. If chosen to generate, the dataset is created using "Generation.C", allowing the user to specify the number, height, and width of images. If chosen not to generate, the dataset is downloaded using the wget method. In both cases, a "images" folder is created where the dataset is placed. This part of the code is managed by the "Gen_files" function. (__fare il wget__)
- _Copying Dataset_:The "images" folder containing the dataset is copied and moved to the "CNN_python" and "TMVA_ML" directories. This code is executed by "move_images_folders" function.
-  _File Presence Verification_: Verifies the presence of the file in various subdirectories.This code is executed by "check_root_file" function. 
-  _Training Environment Selection_: Asks the user to choose whether to start training with ROOT or Python. Then, the user can decide if they want to experiment with the other file as well. This is managed in the main script through keyboard commands.

---vedere forma finale-----

## Analysis.py
SCRIVERE COME RUNNARLO, OKAY LO LASCIAMO SEPARATO MA ALLORA GIUSTIFICHIAMO E IMPLEMENTIAMO MEGLIOOOOOO e DIRE IN CHE CARTELLA METTIAMO I PLOT

This code implements some function to analyse and visualize the input data; several functions are implemented:
- _plot_images(num_images_to_plot, signal_data, background_data)_: takes as argument the number of images that the users wants to plot, the signal and background data; it is useful to visualize some images of the dataset, comparing signal and background
- _pixel_intensity_distribution (sgn_mean,bkg_mean)_: it takes as arguments two arrays containing the mean pixel intensities; this value are used to obtrain the histograms of the intensity distribution. It is useful to understand how pixel are distributed within the images and to look for differences among signal and background data
- _plot_cluster_histogram(cluster1, cluster2)_: it shows a histogram of clusters coming from the KMeans clustering algorithm; it takes as arguments two arrays containign the label of the cluster for each image. It is useful to look for similarities within the data that creates some substractures, called clusters; within clusters, data are considered homogeneous according to a metric defined by the clustering algorithm, in this case the euclidean distance amogn points. In particular, it is helpful to distinguish if the two classes are grouped differently and to simplify data analysis and understanding
- _plot_cluster_centers(centers1, centers2)_: it takes as arguments the centroids for signal and background clusters, obtained by the clustering algorithm as the mean representation of points within a cluster; it is helpful to focus on the principal features of data
- _plot_pixel_distribution(signal_image, background_image)_: this function shows the distribution (the histogram) of pixel within the classes, together with the pixel correltation; it looks for pixel correlation and helps to understand differences in the intensity of images
- _plot_intensity_profile(image_data1,image_data2, axis='row')_: it takes as argument the signal and background arrays; it is the visualization of the intensity profile of images along rows or columns, looking for differnces between signal and background alogn different directions

The resulting plots show no significant differences between signal and background events; the pixel distribution and the pixel intensity distribution have a comparable behaviour, with some differences due to the intrinsic nature of the data. The same can be observed for the intensity profile and the cluster histograms: the two classes are distinguishable, but there is no bias in the distributions that could have affected the training, resulting in an overly simple classification.

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

Alternatively, you can run individual files. First, ensure you have a dataset: yuo have to open root within the "ROOT_Gen" directory and run the Generate_Images.C in the fllowing way:

$ root

$ .L Generate_Images.C

$ Generate_Images (100000, 16, 16)

Once you have the dataset, move it to the "Python_code" and "TMVA_ML" directories. Now you can execute the following commands:

Inside "TMVA":

$root

$ .L TMVA_Classification

$ TMVA_Classification()

Inside "Python_code":

$ python3 Program_start.py

## Only dockerfile

If you prefer to run the dockerfile directly, follow these instructions:

$


