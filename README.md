# S-C Project description
The project compares machine learning techniques for classifying signal and background images, utilizing two main approaches: one based on the TMVA (Toolkit for Multi-Variate Analysis) packages and the other on Python frameworks such as TensorFlow-Keras, PyTorch, and XGBoost for BDT. In particular, the comparison includes Convolutional Neural Network (CNN), Deep Neural Network (DNN), and Boosted Decision Trees (BDT) algorithms. For TMVA, these algorithms are employed directly, while in Python, CNN is implemented using both Keras-TensorFlow and PyTorch, BDT uses XGBoost classifier, and DNN classification is based on Keras-TensorFlow methods. The models are evaluated based on precision, accuracy, F1-score, and ROC curve; also the training time is monitored.

The results are obtained using a dataset of 16x16 images, containing 100,000 images each for signal and background (totaling 200,000 images). This dataset is generated with parameters such as: 100,000 events (n), image height and width of 16 bins each (nh = nw = 16), and each bin is filled with 10,000 events (nRndmEvts). The signal and background distributions have a 5% difference in standard deviation (delta_sigma), and Gaussian noise of 5 is added to each bin of the signal (pixelNoise). Within each bin, data are generated using Gaussian distributions centered on the values obtained from the fits of two functions, with additional Gaussian noise.

In Python, the dataset undergoes normalization of pixel data to a range of 0 to 1, label creation for signal and background (assigned numbers 0 and 1 respectively), and Principal Component Analysis (PCA) for dimensionality reduction. This preprocessing is applied to neural network models, while BDT utilizes normalized data with labels for signal and background without PCA. 

The major aim of this project is to perform a comparison between the TMVA tutorials "CNN_Classification.C" available on the ROOT website at the link: 'https://root.cern/doc/master/TMVA__CNN__Classification_8C.html' and the Python script; the original codes contains also a PyKeras and PyTorch classification. This part was extracted and re-written in order to apply what learned during the lessons and to improve my knoledge about machine learning tecniques. The TMVA code was sliglthly modify and taken as a reference to build the Python analysis, which relies on the same methods implemented in the ROOT script. The implementation of the method is surely different passing from root to python environment, and this affect also performances (see later the paragraph about the results). The results of the comparison between different tecnhiques both for Python and TMVA analysis are presented in the "__Classification_Results__" folder.


## Principal Component Analysis description

Principal Component Analysis is used for both the TMVA and the Python analysis: it consists of a data transformation aimed at reducing the components of each image. To be more precise, each image is composed of 256 components, but not all of them are relevant for classification; the PCA transformation reduces the number of components in each image by finding the directions along which the features show the greatest variations. It helps to reduce the complexity of the data and at the same time improves the performance of the model; it is usually applied to high resolution images and it is also useful to avoid correlations between variables: the CNN classification seems to be more effective when the variables are not correlated. In the TMVA script, the PCA is entered directly by setting "Transformations=P" in the Factory object; in the Python analysis, the PCA is implemented in both the Keras and PyTorch models for the CNN, which requires more complex model definitions; moreover, these two models require a complex shape of the data, two-dimensional for the former and tensors for the latter. PCA transforms this shape into a one-dimensional array, which is easier to manage. 

CNN is a classification tecnhique very suitable for images or grid data, but it is specific and the model is difficult to be build. On the contrary, DNN and BDT are simpler but more robust models with a one-dimensional input: the PCA is not applied to BDT since it is shows good performances instead and so it is chosen as a reference; DNN and CNN intead shows a steep increase in performance after PCA implementation, and they are alsd faster.
The BDT and DNN models are more generic in their application, and their advantages are flexibility and ability to recognise patterns for the former, easy interpretation and robustness for the latter. The DNN are made up of different layers of artificial neurons, while the BDT is based on a decision tree, trying to correct errors made by the previous tree, continuing the training in this way.

For further details and explanations regarding the code implementation, please refer to the specific section.

# Folders organisation 

The project structure includes:

- A folder named "__ROOT_Gen__" that holds a file named "Generation.C". This file is responsible for generating the dataset of images which are saved into the "__images__" subfolder to be used. For more details on dataset generation, refer to the dataset section. If the user has no ROOT available, a backup dataset is provided to run the main analysis with a dockerfile; the generation of the dataset via dockerfile takes long time, thus I provided a reduced dataset for this case.
- Another folder named "__Python_code__", which encompasses several subfolders and generates an additional folder post-code execution: "__plot_results__" which aids in visualizing training parameters alongside the Receiver Operating Characteristic (ROC) curve. This curve illustrates signal efficiency versus background rejection for each model. The "plot_results" folder also contains a comparative visualization of ROC curves and a table showcasing performance metrics such as f1 score, accuracy, precision, and training time across different models. This folder contains also a dockerfile in case users does not have Python installed or an older version; the dockerfile provides itself the right dependences to run the project.
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
- Inside the project, there's also a folder named "__TMVA_ML__". This folder contains a file named "TMVA.C". When executed with the provided bash file (see below), it will generate an additional folder named "images". This folder is generated using "Generation.C" and contains the dataset created. The methods implemented within "TMVA_Classification.C" include a CNN, DNN, and BDT. This file originates from an example provided in the ROOT tutorials, accessible via the following link: 'https://root.cern/doc/master/TMVA__CNN__Classification_8C.html'. This folder contains also a dockerfile to run the analysis if ROOT is not availbale.  More details on the dockerfiles are given in the "__How to run__" section.
- An executable bash file called '__Script.sh__' which serves as a tool for project setup and management. It automates various tasks such as project cleanup, verification of ROOT and Python installations, library compatibility checks, and dataset generation. It allows users to select their preferred training environment (ROOT or Python)
- the folder "__TMVA_plot__" contains some plots automatically originated by the "TMVA_Classification.C" macro; it include the trainign history, the BDT construction scheme, the signal probability distribution and the PDF for CNN and DNN. A further description is provided in the respective readme file of the section.
- the "__Classification_results__" folder contains the final plots displaying the ROC curves for TMVA and python analysis for all models, making thus a comparison between them; it containes also the individual ROC curve for each python model together with a plot of the training history. A table with all the metrics evaluated for each model, together with the training time, is also present; the better performances are coloured in yellow.
- the "__analysis_results__" folder contains:
    -  the plot obtained by running the "Analysis.py" script; when the code is execute, they are saved in "__analysis_plots__" folder, which is deleted every time the bash file is run; here they are reported in the main folder to be commented in the readme file.
    -  the "Analysis.py" script performs some analysis on the input images, as the pixel intensity distribution and the correlation between pixels; more details are reported below
    -  "__DataPreparation__" folder, which manages the load and normalise functions, as well as a reshape function using "Data_Preparation.py".
    -  A dockerfile to set the environment.
- the "__images_backup__" folder contains a backup dataset of 10000 events; it is impossible to upload the whole dataset of 100000 events for the large dimension. It is necessary in the case the users has no ROOT installed and uses dockerfiles. 

Within this folder, plots and results have been described and compared; in addition, each folder contains descriptions of the code implemented and model presentations to help the reader navigate through the project.


# Versions used and packages required
This project was tested on macOS [Sonoma 14.1.2] (M2 chip) with:
- ROOT version: 6.30.06
- Python version: 3.11.5
  - Pandas 2.2.1
  - Numpy 1.26.4
  - Torch 2.2.2
  - Tensorflow 2.16.1
  - Keras 3.3.3
  - Scikit-learn 1.4.2
  - Matplotlib 3.7.2
  - Xgboost 2.0.3
  - Uproot 5.3.2

The overall requirements are reported in the "requirements.txt" file in the folder "__Docker_Folder__".

## Script.sh 
The following bash file allows the user to interactively select how to run the program depending on what is available on his computer. If the user has neither Root nor Python installed, then he will be asked to use dockerfiles; if he has only one of the two, he can run one analysis using the script, and the other using docker. If he has both installed instead, he can choose whether to run the analysis in python or root first. The description of the functions present is done below in detail:
- _Project Cleanup_: Removes directories from previous runs, including "ROOT/images", "Python_code/images", "Python_code/plot_results", and all "pycache" folders within "Python_code" and its subdirectories. This task is handled by the "__generate_delete_command()__" function and "__delete_all_pycache_folders()__".
- _ROOT and Python Verification_: Checks for the presence of ROOT and Python on the computer asking the user which of the two software he has installed on his pc. In case the user has both root and python, then he can run the analysis with both from the bash file; in case he has only one, he can use the bash file to run one analysis and use the dockerfile to run the missing one. In case he has neither software installed, he will be referred back to the dockerfile. This check is performed by the function "__check_software()__"
- _Dataset Generation/Backup_: the user has to decide whether use the backup dataset or to generate it (it can take some time); tis is done by the "__Gen_files()__" fucntions. __Note:__ the code is optimised to run with 100000 events and in this case the better performances are obtained, especially for the stability of the training (looking at the training history); however, the dimension of this file is too large to be managed by upload on github, so I chose a dataset of only 10000 events. __In any case, always check that the filename is correct in both the Python code and the TMVA code__.
- _Copying Dataset_:The "images" folder containing the dataset is copied and moved to the "CNN_python" and "TMVA_ML" directories. This code is executed by "__move_images_folders()__" function.
-  _File Presence Verification_: Verifies the presence of the file in various subdirectories.This code is executed by "__check_root_file()__" function. 
-  _Training Environment Selection_: through keyboard commands, the user can choose whether to run the analysis in root or pyton (or both, depending on one's availability). This is done by the "__main()__" function, which prints to the screen the versions of python and root with which the code was tested, but also the python libraries needed and their version.



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

Alternatively, you can run individual files. First, make sure you have a dataset.

1) Using backup dataset: the user finds a backup dataset in the root folder of the repository ("__images_backup__" folder). Then user needs to put "images_data_16x16_10000.root" into a folder called "images" and move in TMVA_ML and Python_code. When finished, the user can run the code:

        $ cp -r images_backup images

        $ cp -r images TMVA_ML/

        $ cp -r images Python_code 

        $ cp -r images analysis_results 

        $ rm -rf images
   
3) Generating dataset: the user could generate its own dataset using the following lines:

   
       $ root -l -q "ROOT_Gen/Generation.C(100000, 16, 16)"

       $ cp -r images TMVA_ML/

       $ cp -r images Python_code 

       $ cp -r images analysis_results 

       $ rm -rf images
       
Then user can run the project:
- Analysis:

        $ cd analysis_results

        $ python3 Analysis.py

        If the user prefers to delete the results, the following line can be executed:

        $ rm -rf analysis_plots
  
- TMVA:
  
        $ cd TMVA_ML

        $ root -l TMVA_Classification.C

        If the user prefers to delete the results, the following line can be executed:

        $ rm -rf dataset

        $ rm TMVA_CNN_ClassificationOutput.root

- Python ML: 

        $ cd Python_code 

        $ python3 Program_Start.py

        If the user prefers to delete the results, the following line can be executed:

        $ rm -rf plot_results

**N.B.: Once the user has completed a section, they should return to the root folder "__S-C__" using the "_cd .._" command.**

## Only dockerfile

If user prefers to run the dockerfile directly, uses the dataset in backup_images repeating the commands done in the previous section (see section "Run individual files" point 1). 

### Python dockerfile

If the user wants to run dockerfile in python, they should go to "__Python_code__" and run docker:

    $ cd Python_code
    
    $ docker build -t <name_image> .

    $ docker run --rm -it <name_image>

    $ python3 Program_Start.py

### ROOT dockerfile

If the user wants to run dockerfile in ROOT, they should go to "__TMVA_ML__" and run docker:

    $ cd TMVA_ML 

    $ docker build -t <name_image> .

    $ docker run --rm -it <name_image>

### Analysis dockerfile 

If the user wants to explore dataset and run dockerfile in python, they should go to "__analysis_results__" and run docker:

    $ cd analysis_results 

    $ docker build -t <name_image> .

    $ docker run --rm -it <name_image>

    $ python3 Analysis.py
    
