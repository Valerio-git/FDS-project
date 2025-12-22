# FDS-project

This is the code for our FDS-project inspired by **Hooly**, the smart bin built by the Italian start-up *Ganiga.ai*.

The structure of the code is inside the src folder.

## Main file
The *main* file contains the entire pipeline to execute the code. 
Firstly it sets the seeds and finds whether there is a GPU to work on. Then it receives instructions on:
- which model to use (CNN or resnet50).
- which dataset to work on (raw dataset, the original one or white dataset).

If the chosen model is CNN, then it trains the model to find the optimal values in the grid search, saving everything in *cnn_stage1_A.pth*. Otherwise (model_type = resnet), it skips this part going directly to the fine tuning phase. 
Then, if the dataset selected is the **white dataset**, it implements training and validation for the fine-tuned model (cnn or resnet). Otherwise, it goes directly to the testing phase of the selected model. 

## Models folder 
It contains:
- *CCN.py* in which there is the implementation of our custom-built CNN.
- *resnet.py* which contains the script for uploading the pre-trained *resnet50* model.

## Data folder
It contains utilities for dataset loading, preprocessing, and management:
- *data_loader.py* defines the `WasteDataset` PyTorch class for raw and white datasets and handles train, validation and test set (60%/20%/20%) with consistent class ordering.
- *transforms.py* defines image transformation pipelines (augmentation for training, normalization for validation and testing) and uses ImageNet statistics for ResNet compatibility.
- *reorganize_datasets.py* reorganizes datasets according to `mapping.json` and moves and renames images from item to category folders.
- *white_dataset.py* creates white-background dataset using `rembg`, copies `default` images and removes background from `real_world` images.

## Training folder
All the codes for the training of the model and its validation are contained in the *training* folder:
- *train_utils.py* contains functions to load and split the dataset, train the model, and validate it
- *train_hyperparameters.py* calls back all the functions in *train_utils.py* and uses them not only to train and validate the model but also to implement a grid search to find the best:
    - learning rate
    - batch size
    - weight decay 

## Fine tuning folder
This folder contains the main files:
- *fine_tuning.py* which contains the functions used during the fine tuning of both CNN and resnet50 on the **white dataset**.
- *Fine_tune_CNN.py* that trains and validates fine-tuned *CNN* on the new white dataset using optimal values (for batch size, learning rate and weight decay) found during the grid search.
- *Fine_tune_resnet.py* which trains and validates fine-tuned *resent50* on the white dataset. 

## Testing folder 
Files for the testing phase are inside this folder:
- *test_utils.py* contains functions to load the test dataset and to evaluate on it. All these functions have a boolean parameter **white** which selects the dataset to operate on. This allows *test_utils.py* to be used for CNN, fine-tuned CNN and fine-tuned resnet50. It also includes functions to visualize images with predictions.
- *test.py* it contains the pipeline for the testing of the model.

## Utils folder
It contains files with different functions:
- *confusion_matrices.py* creates the confusion matrix for each model.
- *function.py* contains general functions for setting the seed (allowing reproducibility of results) and for interacting with the console (choosing which model to use and whether or not to do fine tuning).
- *plots.py* contains functions for plotting training values of **accuracy**, **F1-score** and **loss** for each model.
- *split_dataset.py* creates a .json file in which training, validation and test set are contained. If the file already exists, it doesn't create another one (this allowed us not to recreate a  splitting each time). Depending on which dataset you're working on, it will create or *splits_raw.json* or *splits_white.json*.
- *data_utils.py*, although it is not inside this folder, it contains the fuctions to find the path for different datasets.

## Checkpoints folder
It contains model files and best parameters found after: training and grid search (*cnn_stage1_A.pth*), training made on fine-tuned models (*cnn_stage2.pth*, *resnet_stage2.pth*). 
