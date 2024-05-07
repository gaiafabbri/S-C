#include <iostream>
#include <string>
#include <cstdlib>
#include <map>

#include "TMVA/TMVAGui.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Factory.h"
#include "TMVA/Tools.h"

#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TObjString.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TChain.h"
/***
 
    # TMVA Classification Example Using a Convolutional Neural Network
 
**/
//-----------------------------------------------------------------------------------
/// @brief Run the TMVA CNN Classification example
/// @param nevts : number of signal/background events. Use by default a low value (1000)
///                but increase to at least 5000 to get a good result
/// @param opt :   vector of bool with method used (default all on if available). The order is:
///                   - TMVA CN
///                   - TMVA DNN
///                   - TMVA BDT
//-----------------------------------------------------------------------------------

void TMVA_Classification(int nevts = 1000, std::vector<bool> opt = {1, 1, 1})
{
 
    //Accessing the input file, printin an error message if the file is not found
   int imgSize = 16 * 16;
   TString inputFileName = "TMVA_ML/images/images_data_16x16_100000.root";
 
   bool fileExist = !gSystem->AccessPathName(inputFileName);
 
   // if file does not exists
    if (!fileExist) {
        std::cout << "error: the file is not present" << std::endl;
    }

//-----------------------------------------------------------------------------------
 //initialization of boolean variables checking if the vector opt contains the methods for classification
// for each variable, the size of the vector is checked, and it must contain one element for the first variable, two elements for the second and so on
   bool useTMVACNN = (opt.size() > 0) ? opt[0] : false;
   bool useTMVADNN = (opt.size() > 1) ? opt[1] : false;
   bool useTMVABDT = (opt.size() > 2) ? opt[2] : false;
//-----------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------
//running with multi-thread option or with GPU
#ifndef R__HAS_TMVACPU
#ifndef R__HAS_TMVAGPU
   Warning("TMVA_CNN_Classification",
           "TMVA is not build with GPU or CPU multi-thread support. Cannot use TMVA Deep Learning for CNN");
   useTMVACNN = false;
#endif
#endif
 
   bool writeOutputFile = true;
 
#ifdef R__USE_IMT
   int num_threads = 4;  // use by default 4 threads if value is not set before
   // it is a variable to switch off multi-thread in OpenBLAS to avoid conflict with Intel Threading Building Blocks(tbb)
   gSystem->Setenv("OMP_NUM_THREADS", "1");
 
   // do enable MT running
   if (num_threads >= 0) {
      ROOT::EnableImplicitMT(num_threads);
   }
#endif
 
   TMVA::Tools::Instance();
 
    //printing the number of threads
   std::cout << "Running with nthreads  = " << ROOT::GetThreadPoolSize() << std::endl;
 
//creating the output root file
    TString outfileName("TMVA_CNN_ClassificationOutput.root");
   TFile *outputFile = nullptr;
   if (writeOutputFile)
      outputFile = TFile::Open(outfileName, "RECREATE");
//-----------------------------------------------------------------------------------
 
    
//-----------------------------------------------------------------------------------
   /***
       ## Create TMVA Factory
 
Create the Factory class. Later you can choose the methods whose performance you'd like to investigate.
The factory is the major TMVA object you have to interact with. Here is the list of parameters you need to pass
 
- The first argument is the base of the name of all the output weight files in the directory weight/ that will be created with the method parameters
- The second argument is the output file for the training results
- The third argument is a string option defining some general configuration for the TMVA session. For example all TMVA output can be suppressed by removing the "!" (not) in front of the "Silent" argument in the
   option string
- note that we disable any pre-transformation of the input variables and we avoid computing correlations between
   input variables
-scatter plot and correlation matrixes for 256 varibales need to much memory and computational efforts to be calculated, so we disable them
   ***/
//-----------------------------------------------------------------------------------
    
   TMVA::Factory factory( "TMVA_CNN_Classification", outputFile, "!V:ROC:!Silent:Color:AnalysisType=Classification:Transformations=P:!Correlations");
    
//-----------------------------------------------------------------------------------
   /***
 
       ## Declare DataLoader(s)
 The next step is to declare the DataLoader class that deals with input variables
Define the input variables that shall be used for the MVA training note that you may also use variable expressions, which can be parsed by TTree::Draw( "expression" )]
In this case the input data consists of an image of 16x16 pixels. Each single pixel is a branch in a ROOT TTree
 
   **/
//-----------------------------------------------------------------------------------
    
   TMVA::DataLoader loader("dataset");
    
//-----------------------------------------------------------------------------------
   /***
       ## Setup Dataset(s)
Define input data file and signal and background trees
   **/
//-----------------------------------------------------------------------------------

   std::unique_ptr<TFile> inputFile{TFile::Open(inputFileName)};
   if (!inputFile) {
      Error("TMVA_CNN_Classification", "Error opening input file %s - exit", inputFileName.Data());
      return;
   }
 
   // --- Register the training and test trees
 
   auto signalTree = inputFile->Get<TTree>("sig_tree");
   auto backgroundTree = inputFile->Get<TTree>("bkg_tree");
 
   if (!signalTree) {
      Error("TMVA_CNN_Classification", "Could not find signal tree in file '%s'", inputFileName.Data());
      return;
   }
   if (!backgroundTree) {
      Error("TMVA_CNN_Classification", "Could not find background tree in file '%s'", inputFileName.Data());
      return;
   }
 
   int nEventsSig = signalTree->GetEntries();
   int nEventsBkg = backgroundTree->GetEntries();
 
   // global event weights per tree (single event weights can also be set)
   Double_t signalWeight = 1.0;
   Double_t backgroundWeight = 1.0;
 
   // Adding signal and background tree containing the events
   loader.AddSignalTree(signalTree, signalWeight);
   loader.AddBackgroundTree(backgroundTree, backgroundWeight);
 
   // add event variables (image): it is an array of 256 variables
   // use new method (from ROOT 6.20 to add a variable array for all image data)
   loader.AddVariablesArray("vars", imgSize);

   // If you want apply additional cuts on the signal and background samples the followinf must be modified: it is useful in case signal and background events are in the same tree
   TCut mycuts = "";
   TCut mycutb = "";
    
//-----------------------------------------------------------------------------------
/// Tell the factory how to use the training and testing events
   //
/// -If no numbers of events are given, half of the events in the tree are used for training, and the other half for testing:
///-nTrain_Signal=0 and nTrain_Background=0 means using all the events for training (and for testing also); specifying the number of events used to train an putting nTestSignal=0 and nTest_Background=0 means using the remaing part of the dataset to test
///- note we disable the computation of the correlation matrix of the input variables as told before
//-----------------------------------------------------------------------------------
    
   int nTrainSig = 0.8 * nEventsSig;
   int nTrainBkg = 0.8 * nEventsBkg;
 
   // build the string options for DataLoader::PrepareTrainingAndTestTree
   TString prepareOptions = TString::Format(
      "nTrain_Signal=%d:nTrain_Background=%d:SplitMode=Random:SplitSeed=100:NormMode=NumEvents:!V:!CalcCorrelations",
      nTrainSig, nTrainBkg);
 
   loader.PrepareTrainingAndTestTree(mycuts, mycutb, prepareOptions);

//-----------------------------------------------------------------------------------
   /***
 
       DataSetInfo              : [dataset] : Added class "Signal"
       : Add Tree sig_tree of type Signal with 10000 events
       DataSetInfo              : [dataset] : Added class "Background"
       : Add Tree bkg_tree of type Background with 10000 events
   **/
//-----------------------------------------------------------------------------------
 
//-----------------------------------------------------------------------------------
   /****
        # Booking Methods:
-the first argument is the predefined enumerator specifying the classifier
-the second argument is an user-defined name
-the third argument is a string containing all the configuration options
   **/
//-----------------------------------------------------------------------------------
 
   // Boosted Decision Trees
   if (useTMVABDT) {
      factory.BookMethod(&loader, TMVA::Types::kBDT, "BDT",
                         "!V:NTrees=200:MinNodeSize=2.5%:MaxDepth=2:BoostType=AdaBoost:AdaBoostBeta=0.5:"
                         "UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20");
   }
//-----------------------------------------------------------------------------------

 //Booking Deep Neural Network
 
   if (useTMVADNN) {
 
      TString layoutString(
         "Layout=DENSE|64|RELU,BNORM,DENSE|64|RELU,BNORM,DENSE|64|RELU,BNORM,DENSE|64|RELU,DENSE|1|LINEAR");// the number of neuron is chosen to maximize the model performance and to not apply a too complex representation of data
 
      // Training strategies
      // one can catenate several training strings with different parameters (e.g. learning rates or regularizations
      // parameters) The training string must be concatenates with the `|` delimiter
      TString trainingString1("LearningRate=1e-3,Momentum=0.9,Repetitions=1,"
                              "ConvergenceSteps=5,BatchSize=100,TestRepetitions=1,"
                              "MaxEpochs=10,WeightDecay=1e-4,Regularization=None,"
                              "Optimizer=ADAM,DropConfig=0.0+0.0+0.0+0.");
 
      TString trainingStrategyString("TrainingStrategy=");
      trainingStrategyString += trainingString1; // + "|" + trainingString2 + ....
 
      // Build now the full DNN Option string
 
      TString dnnOptions("!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=None:CreateMVAPdfs:"
                         "WeightInitialization=XAVIER");
      dnnOptions.Append(":");
      dnnOptions.Append(layoutString);
      dnnOptions.Append(":");
      dnnOptions.Append(trainingStrategyString);
 
      TString dnnMethodName = "TMVA_DNN_CPU";
       
// use GPU if available
#ifdef R__HAS_TMVAGPU
      dnnOptions += ":Architecture=GPU";
      dnnMethodName = "TMVA_DNN_GPU";
#elif defined(R__HAS_TMVACPU)
      dnnOptions += ":Architecture=CPU";
#endif
 
      factory.BookMethod(&loader, TMVA::Types::kDL, dnnMethodName, dnnOptions);
   }
 
//-----------------------------------------------------------------------------------
   /***
    ### Book Convolutional Neural Network in TMVA
 
    For building a CNN one needs to define
-  Input Layout :  number of channels (in this case = 1)  | image height | image width
-  Batch Layout :  batch size | number of channels | image size = (height*width)
Then one add Convolutional layers and MaxPool layers.
 
For Convolutional layer the option string has to be:
- CONV | number of units | filter height | filter width | stride height | stride width | padding height | paddig
   width | activation function
- note in this case we are using a filer 3x3 and padding=1 and stride=1 so we get the output dimension of the
   conv layer equal to the input
- note we use after the first convolutional layer a batch normalization layer. This seems to help significantly the
   convergence
    
 For the MaxPool layer:
- MAXPOOL  | pool height | pool width | stride height | stride width
 
The RESHAPE layer is needed to flatten the output before the Dense layer
Note that to run the CNN is required to have CPU  or GPU support
   ***/
//-----------------------------------------------------------------------------------
 
   if (useTMVACNN) {
 
      TString inputLayoutString("InputLayout=1|16|16");
 
      // Batch Layout
      TString layoutString("Layout=CONV|10|3|3|1|1|1|1|RELU,BNORM,CONV|10|3|3|1|1|1|1|RELU,MAXPOOL|2|2|1|1,"
                           "RESHAPE|FLAT,DENSE|64|RELU,DENSE|1|LINEAR");
 
      // Training strategies.
      TString trainingString1("LearningRate=1e-3,Momentum=0.9,Repetitions=1,"
                              "ConvergenceSteps=5,BatchSize=100,TestRepetitions=1,"
                              "MaxEpochs=10,WeightDecay=1e-4,Regularization=None,"
                              "Optimizer=ADAM,DropConfig=0.0+0.0+0.0+0.0");
 
      TString trainingStrategyString("TrainingStrategy=");
      trainingStrategyString +=
         trainingString1; // + "|" + trainingString2 + "|" + trainingString3; for concatenating more training strings
 
      // Build full CNN Options.
      TString cnnOptions("!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=None:CreateMVAPdfs:"
                         "WeightInitialization=XAVIER");
 
      cnnOptions.Append(":");
      cnnOptions.Append(inputLayoutString);
      cnnOptions.Append(":");
      cnnOptions.Append(layoutString);
      cnnOptions.Append(":");
      cnnOptions.Append(trainingStrategyString);
 
      //// New DL (CNN)
      TString cnnMethodName = "TMVA_CNN_CPU";
       
// use GPU if available
#ifdef R__HAS_TMVAGPU
      cnnOptions += ":Architecture=GPU";
      cnnMethodName = "TMVA_CNN_GPU";
#else
      cnnOptions += ":Architecture=CPU";
      cnnMethodName = "TMVA_CNN_CPU";
#endif
 
      factory.BookMethod(&loader, TMVA::Types::kDL, cnnMethodName, cnnOptions);
   }
//-----------------------------------------------------------------------------------
   
   ///  ## Train Methods
 
   factory.TrainAllMethods();
 
   /// ## Test and Evaluate Methods
 
   factory.TestAllMethods();
 
   factory.EvaluateAllMethods();
    // Dopo factory.EvaluateAllMethods();
 
   /// ## Plot ROC Curve
 
    auto c1 = factory.GetROCCurve(&loader);
    c1->Draw();
     //c1->Print("arturo.pdf");
  
    // close outputfile to save output file
    outputFile->Close();
    if (!gROOT->IsBatch()) TMVA::TMVAGui( outfileName );
 }
