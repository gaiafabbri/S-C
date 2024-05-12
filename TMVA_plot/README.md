# TMVA Analysis Results:
Some TMVA plots are reported in the "__TMVA_plot__" folder; they are obtained by running "TMVA_Classification.C". The macro automatically creates a folder called "__dataset__" containing the "__weights__" folder with the weight files for each method booked by TMVA, and the "__plots__" folder; they are deleted by the bash file wherever it is run, but some examples are reported. Among them:
- The **overtraining test** evaluates the response of the model to the test sample; it measures the ability of the model to generalise the prediction to an unknown dataset. It looks at the distribution of training and test data, and if the model is learning correctly, the two distributions should look similar. The Kolmogorov-Smirnov test quantifies the differences between the two distributions and returns a value D, which must be as small as possible, indicating that the two distributions are close; if this value is small, the model is correctly trained and the probability of overtraining is reduced. For the models implemented, the D values are small (from 0.047 for the BDT to 0.451 for the DNN for signal, for example) and the test and training data distributions overlap, indicating that there is no overtraining;

<p align="center">
  <img width="450" height="300" src="https://github.com/gaiafabbri/S-C/blob/main/TMVA_plot/overtrain_TMVA_CNN_CPU.png">
  <img width="450" height="300" src="https://github.com/gaiafabbri/S-C/blob/main/TMVA_plot/overtrain_TMVA_DNN_CPU.png">
</p>
  
- The **response distribution** of the models: the DNN and the CNN show two well-separated distributions, peaked at 1 for signal and at 0 for background, indicating that the models separate the two classes well; the BDT, on the other hand, shows two more overlapping distributions, indicating that the model does not perform in the same way, as the ROC curve has already shown. The response distribution in this case peaks around 0.1 for signal and around -0.1 for background, so the two peaks are separated but not absolutely. The different values of the peaks may also indicate a different learning strategy with respect to neural networks.

<p align="center">
  <img width="400" height="280" src="https://github.com/gaiafabbri/S-C/blob/main/TMVA_plot/mva_BDT.png">
  <img width="400" height="280" src="https://github.com/gaiafabbri/S-C/blob/main/TMVA_plot/mva_TMVA_CNN_CPU.png">
  <img width="400" height="280" src="https://github.com/gaiafabbri/S-C/blob/main/TMVA_plot/mva_TMVA_DNN_CPU.png">
</p>
  
- The **distribution of the signal probability** for the DNN and for the CNN: in both cases, the signal probability has a peak around zero for the background images and a peak around one for the signal images; however, the peak for the signal is smaller and the signal probability is present throughout the entire interval, indicating that the models have some difficulties in accurately stating when the signal is actually present.

<p align="center">
  <img width="450" height="300" src="https://github.com/gaiafabbri/S-C/blob/main/TMVA_plot/proba_TMVA_CNN_CPU.png">
  <img width="450" height="300" src="https://github.com/gaiafabbri/S-C/blob/main/TMVA_plot/proba_TMVA_DNN_CPU.png">
</p>

- The **distribution of signal rarity** for the DNN and the CNN show similar results to the signal probability distribution; for both models the peak around zero for background is present and the peak around one for signal is higher with respect to the signal probability. It is a measure of the rarity of the signal with respect to the background, so the distinction between the two classes is more pronounced with respect to the signal probability; it can be interpreted as the fact that around one the models are more certain to find signal, but the presence of signal in the whole interval between 0 and 1 indicates that the models are not sure in the classification of the two classes.

  <p align="center">
  <img width="450" height="300" src="https://github.com/gaiafabbri/S-C/blob/main/TMVA_plot/rarity_TMVA_CNN_CPU.png">
  <img width="450" height="300" src="https://github.com/gaiafabbri/S-C/blob/main/TMVA_plot/rarity_TMVA_DNN_CPU.png">
</p>

- The **inverse of the background rejection as a function of the signal efficiency** is shown; it represents the ability of the model to detect and reject the background. It must peak at low values of signal efficiency, indicating that the models perform well in discriminating the signal when the background rejection is high (and so the curve goes to zero).

<p align="center">
  <img width="500" height="380" src="https://github.com/gaiafabbri/S-C/blob/main/TMVA_plot/invBeffvsSeff.png">
</p>

They are obtained by adding to the macro the line "TMVA::TMVAGui( outfileName )": in this way the root file "TMVA_CNN_ClassificationOutput.root" is given as input to some macros that are automatically executed by the user on a graphical canvas; once the root code is executed, the Gui menu is opened and by simply pressing the selections the plot is displayed and saved.
