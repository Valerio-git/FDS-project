# FDS-project

This is the code for our FDS-projects inspierd to **hooly** the smart bin built by the italian start-up *Gaina.ay*.

The structure of the code it's inside the src folder.

## Main file
The file *main* contains the entire pipeline to execute the code. 
Firstly it sets the seeds and find whether there is a GPU to work on. Then it will recive instrunction on:
- which model to use (CNN or resnet50)
- which dataset to work on (raw dataset, the original one or white dataset)
If the model chosen is CNN then it will train the model finding the optimal values in the grid search, saving everything in *cnn_stage1_A.pth*. Otherwise (model_type = resnet it will skip this part going strictly to fine tuning phase). 
Then if the dataset selected was **white dataset** then it will implement the training and validation for fine tune model (cnn or resnet). Otherwise it will go directly to the tesing phase of the model selected. 

## Models folder 
It contains:
- *CCN.py* in which there's the implememtation of our custom-built CNN.
- *resnet.py* which contains the script for updloading the pre-trained model *resnet50*.

## Training folder
All the codes for the training of the model and its validation are contained in the folder *training*
- *train_utils* contains the functions required to load and split the dataset, the functions for training the model and for validating it
- *train_hyperparameters.py* calls back all the functions in *train_utils.py* and uses them not only to train and validate the model but also to implement a grid search to find the best:
    - learning rate
    - batch size
    - weight decay 

## Fine tuning folder
This folder contains the main files:
- *fine_tuning.py* which contains the function used during the fine tuning of both the CNN and resnet50 on the **white dataset**.
- *Fine_tune_CNN.py* that trains the and validates finetuned *CNN* on the new white dataset using as optimal values for batch size, learning rate and weight decay the ones found during the grid search.
- *Fine_tune_resnet.py* which trains and validates finetuned *resent50* on the new whitedates 

## Testing folder 
Files for the testing phase are inside this folder:
- *test_utils.py* contains functions to load the test dataset and to evaluate on it. All these functions have a boolena parameter **white** which selectes the dataset to operate on. This allows *test_utils.py* to be used both for CNN and for fine-tune CNN and fine tuned resnet50. Then inside there are also functions used for the representation of some images with the prediction made on them.
- *test.py* it contains the pipeline for the testing of the model

## Utils folder
It contains file whit different functions:
- *confusion_matrices.py* creates the confusion matrix for each model
- *function.py* contains general function for setting the seed (allowing reproducibility of results) and for interacting with the console (choosing which model to use and whether or not to do fine tuning).
- *plots.py* contains functions for plotting training values of **accuracy**, **F1-score** and **loss** for each model.
- *split_dataset.py* it's called during the splitting of the dataset. It creates a .json file in which train, validation and test set are contained. If the file already exists it will not create another one. (This allowed us to not recreate each time a splitting). Cleary depending on which dataset we're working on it will creates *splits_raw.json* or *splits_white.json*.
- *data_utils.py* although it is not inside this folder it contains the fuctions to get the path for the different datasets.

**Checkpoints folder**
It contains files in which there are the models and the best parameters found after: training and grid search (*cnn_stage1_A.pth*), the training made on fine tuned models (*cnn_stage2.pth*, *resnet_stage2.pth*). 
