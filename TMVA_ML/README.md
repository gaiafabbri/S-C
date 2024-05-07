## TMVA_Classification.C 

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
