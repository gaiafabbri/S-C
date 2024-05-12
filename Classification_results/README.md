
# Comments on results obtained

## TMVA results:
For the TMVA analysis, the most performing model seems to be the CNN, followed immediately by the DNN; the BDT semms a little worst in classification. The ROC-integer for each model is respectively 0.941, 0.937 and 0.821, with the CNN which shows also the better signal efficiency (0.955, 0,950 and 0.781 for the 30% of background, respectively). CNN and DNN show very similar behaviour, while the BDT model is a little more discriminative, but still performs an efficient classification. However, the BDT shows a fast training, with 32 s for training and 0.65 s for evaluation; the DNN is also faster, with 17 s for training and 1.2 s for evaluation. The CNN is significantly slower, with 170 s for training and 8 s for evaluation.

<p align="center">
  <img width="500" height="380" src="https://github.com/gaiafabbri/S-C/blob/main/Classification_results/TMVA_ROC.png">
</p>

## Python results:
As far as the Python analysis is concerned, the results between the different models are very similar and all the models seem to perform quite well in the classification. The Torch CNN is the one with the higher accuracy and precision, while the DNN has the better F1 score; however, the values for the other model are not far from the optimum and in general they are about 86% for all the models. Again, the CNN with Keras has the slowest training time of 66 s, but the other models take 11 s, 5 s and 21 s for Torch CNN, BDT and DNN respectively. Confusion matrices were also computed to evaluate the performance of the classifiers: on the test dataset of 40,000 events, all models show about 17,000 true positives and true negatives, for a total of 34,000 correctly identified events; only the BDT performs slightly worse, with about 16,000 true positives and 16,000 true negatives. The ROC curves obtained from the Python analysis are quite consistent with the TMVA curves, as the CNN (both with Keras and PyTorch) and DNN curves almost overlap, while the BDT curve is slightly lower.

<p align="center">
  <img width="550" height="380" src="https://github.com/gaiafabbri/S-C/blob/main/Classification_results/Comparison_among_models_100000_16x16.png">
  <img width="620" height="340" src="https://github.com/gaiafabbri/S-C/blob/main/Classification_results/Results.png">
</p>

For each model, together with the ROC curve, also the training plot is shown; for the DNN and the CNN both with Keras and PyTorch the loss and the accuracy are monitored both on the training and test dataset. For the BDT only the loss for training and test is reported

<p align="center">
  <img width="500" height="220" src="https://github.com/gaiafabbri/S-C/blob/main/Classification_results/BDT_100000_16x16.png">
  <img width="500" height="220" src="https://github.com/gaiafabbri/S-C/blob/main/Classification_results/CNN with tensorflow-keras_100000_16x16.png">
</p>

<p align="center">
  <img width="500" height="220" src="https://github.com/gaiafabbri/S-C/blob/main/Classification_results/CNN with torch_100000_16x16.png">
  <img width="500" height="220" src="https://github.com/gaiafabbri/S-C/blob/main/Classification_results/DNN_100000_16x16.png">
</p>


## Conclusion:
In conclusion, CNNs are generally preferred for image classification due to
- Their spatial invariance, which means that they are able to identify patterns even if they do not have the same position within the image.
- The use of more layers makes the CNN suitable for learning complex patterns, since the first layers learn the simpler patterns, which are then used to learn more complex representations.

Keras provides a user-friendly environment with predefined layers, making it easy for less experienced users to construct high-performance networks; PyTorch is more flexible, as the computational design of the network is defined dynamically and modified during the program run. However, it seems to be more complicated to implement. BDT and DNN are more flexible and generic models that seem to perform quite well in classifying background and signal images anyway: they are a suitable choice for image classification if the dataset is not too complicated and they provide easily interpretable results with reduced training time. In general, CNN are more complex models to train, but they can lead to very high performances; however, for our dataset, CNN leads to a slower training time without a significant improvement in the performances, especially with respect to the DNN model.
