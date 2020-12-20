# Parkinson's Disease Prediction

> Parkinson’s Disease Prediction from Speech Analysis, Machine Learning Project at Indraprastha Institue of Information Technology, Delhi

This repository contains the python code for training and testing multiple models on the [UCI ML Repository's Parkinson's Disease Classification Data Set](https://archive.ics.uci.edu/ml/datasets/Parkinson%27s+Disease+Classification). The aim of this project was to analyse the performance of different sklearn models on their ability to classify if a patient has Parkinson's Disease or not from the speech signal analysis of the patient.  


## Running Pre-trained Models
>(run_model.py)
1. Use `python run_model.py --help` to see the complete list of trained models. 
2. Choose a model with the `python run_model.py --model {model_number}` to use the pretrained model on the parkinsons's disease dataset.
   
## Everything from scratch
1. `EDA.py` to see the data analysis visual plots.
2. `generate_weights.py` to save weights for models with best hyperparameters ([see more](##Generating-Weights))
3. `run_model.py` to see results on the saved weights.


## File Descriptions
| Name                 | Descripton                               |
| ---------------------|:----------------------------------------:| 
| run_model.py         | Run Model from pre-trained weights       | 
| best_params.py       | Get Models for best hyperparameters      |   
| generate_weights.py  | Save Weights for best models             | 
| pre_processing.py    | PCA, Feature Selection, train-test split |
| EDA.py               | Exploratory Data Analysis on datas       |


## Generating-Weights
>(generate_weights.py)
1. Use ```python run_model.py --help``` to see the complete list of trained models. 
2. Choose a model with the `python run_model.py --model {model_number}` to use the pretrained model on the parkinsons's disease dataset.   