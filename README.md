# FDS-project

This is the code for our FDS-projects inspierd to **hooly** the smart bin built by the italian start-up *Gaina.ay*.

The structure of the code it's inside the src folder.

## File main
The file *main* contains the entire pipeline to execute the code. 

## Folder models 
It contains:
-  CCN.py in which there's the implememtation of our custom-built CNN.
- resnet.py which contains the script for updloading the pre-trained model *resnet50*.

## Folder training
All the codes for the training of the model and its validation are contained in the folder *training*
- *train_utils* contains the functions required to load and split the dataset, the functions for training the model and for validating it
- *train_hyperparameters.py* calls back all the functions in *train_utils.py* and uses them not only to train and validate the model but also to implement a grid search to find the best:
    - learning rate
    - batch size
    - weight decay 

## Folder fine tuning
This folder contains the main file *fine_tuning.py* which contains the function used during the fine tuning of both the CNN and resnet50 on the **white dataset**.
*Fine_tune_CNN.py* trains the and validates finetuned *CNN* on the new white dataset using as optimal values for batch size, learning rate and weight decay the ones found during the grid search.
*Fine_tune_resnet.py* trains the and validates finetuned *resent50* on the new whitedates 
