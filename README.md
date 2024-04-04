# S-C
The following project aims to classify signal and background images using machine learning techniques; in particular, a comparison is made between TMVA (Toolkit for Multi-Variate Analysis) packages and Python libraries. The former includes CNN, DNN and Boosted Decision Tree (BDT); the latter implements the same algorithms: CNN is permormed with both keras-tensorflow and pytorch, BDT uses xgboost classifier, while DNN classification is based on keras-tensorflow methods. The models are evaluated in terms of precision, accuracy, f1-score and roc curve.

# Main files description
The project is mainly composed by the following codes:
1) images_generation.C is a root macro to generate images width*heigt contained in a root file; images are generating filling 2d histograms with Gaussian distribution; each images is made by nRndmEvts.Two different histograms are filled according to two bidimensional gaussian distribution f1 and f2 specified by different parameters; they constitute the signal and the background, respectively. In each bin, noise is added; then, the signal and the background trees are filled and the root file is edited
2) Classification.py: it is the main python code; the data pre-processing is performed here, extracting variables form the signal and background tree respectively and normalizing the data. Then the user is asked to choose a model to perform the classification among BDT, DNN; CNN with keras-tensorflow or pytorch; according to the dimensionality of the dataset, the analysis is performed including Principal Component Analysis (PCA).
3) The "DataPreparation" folder contains all the preprocessing files: the exact explanation is given in the comments inside the files.
4) The "Models" folder contain the defintion of the models used to train the dataset; more precise explanation are reported as comments

   AGGIUNGERE LE COSE MANCANTI



Qui di seguito, vengono mostrati i risultati ottenuti sotto le impostazioni di default 16x16 e distribuzione Gaussiana:


# Organisation of Files
The files are organized as follows:
-The main folder contains Classification.py and TMVA.C 
-....

The project contains:

- Una cartella denominata "Code" che contiene seguenti file:
    - "TMVA.C" che realizza le seguenti tecniche di machine learning: BDT, CNN_TMVA_CPU, DNN_TMVA_CPU (per ulteriori chiarimenti guardare la sezione successiva circa le modifiche)
    - "Keras.py" che realizza implementa una CNN per l'addestramento sui dati usando Keras
    - "PyTorch.py" anche in questo caso viene implementata una CNN, ma questa volta usando Pytorch
- Una cartella denominata "Data Generation", che contiene:
    - "Gen_Data.C" per la generazione del dataset, la funzione al suo interno costruisce delle immagini sulla base delle indicazioni che vengono passate dall'utente al momento dell'esecuzione dell'eseguibile "run.sh"
    - Contiene anche i file di output della funzione di "Gen_Data.C" solitamente salvati come segue: [formato_immagine].root, per ulteriori informazioni guardare la sezione relativa al commento del codice.
- Un eseguibile chiamato "run.sh" che ha il compito di:
    - Assistere l'utente nella generazione di immagini
    - Verificare la presenza di ROOT e Python sul computer
    - Verificare la presenza delle librerie necessarie all'esecuzione del processo di verificare la presenza e la compatibilità tra le librerie di Python ed eventualmente effettuare aggiornamenti
    - Permettere all'utente di scegliere se eseguire un codice alla volta, ed eventualmente il codice che si preferisce eseguire per primo, o eseguirli tutti in serie
    - Gestisce il salvataggio dei risultati
    - Quando eseguito effettua una operazione di pulizia dei tentativi precedenti
    - Se non disponibile ROOT e/o Python, l'eseguibile indirizzerà l'utente all'utilizzo di un dockerfile
    - Se richiesto, viene resa possibile l'apertura dei pdf creati come risultati degli output (?)
- Il "dockerfile" che è stato costruito nel caso in cui l'utente non abbia ROOT e/o Python. Esso crea un ambiente per poter usare sia ROOT, sia Python indipendentemente dalla presenza di uno o dell'altro sul computer. In termini pratici sostituirà l'eseguibile "run.sh" replicandone i compiti e la gestione del progetto.
- Una cartella denominata "link", contiene i seguenti link ad una codice scritto in Coolab che ripropone i metodi usati nei vari algoritmi, però implementando l'uso della GPU di coolab, in questo caso sarà possibile scegliere due approci: o usare un file di default importato tramite wget da dropbox oppure diversamente è possibile scegliere di crearsi un dataset, ma in questo caso l'utente dovrà caricare il file sul proprio google drive e partire dalla casella di montaggio del drive.
    - [Link1]
    - [Link2]
- Una cartella denominata "Output" che contiene, se seguenti subfolders:
    - [sub1]
    - [sub2]
    - [sub3]
    - metti un pdf che metta in tabella i risultati delle accuracy nella valutazione dei vari modelli [se po' fa?]
- Una cartella che contiene un codice per analizzare i dati generati denominata "File_image"
- Una cartella chiamata "Default" che contiene tutto il codice ma con un file di dati di default che proviene da dropbox e scaricato per mezzo di wget method
 




# Versioni usate e pacchetti richiesti
This project was tested on macOS [Versione] Sonoma (M2 chip) with:
- ROOT version:
- Python version:
  - Pandas
  - Numpy
  - Torch
  - Tensorflow
  - Keras
  - Scikit-learn
  - ...

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

Una volta che hai fatto:

$ git clone https://github.com/gaiafabbri/S-C.git 

Andare al percorso del progetto:

$ cd your/path/to/S-C

Ottenere i permessi per l'avvio dell'eseguibile:

$ chmod +x run.sh

Esguire il file bas:

$ ./run.sh

