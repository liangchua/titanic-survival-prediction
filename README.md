![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)
![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

# TITANIC SURVIVAL PREDICTION

This is a Kaggle competition for beginner - The Legendary Titanic Machine Learning Competition. 

[Link to the competition](https://www.kaggle.com/c/titanic "Titanic Compatition")


### THE CHALLENGE

Build a predictive model to solved the problem "What sorts of people were more likely to survive?"


### THE OBJECTIVE

Predict whether a Titanic's passenger survived 


### THE DESCRIPTION OF UPLOADED FILES 

|File Name|Description|
|---------|-----------|
|Titanic_1.jpg|A Titanic image|
|minmax.pkl|The fitted MinMaxScaler model (refer to line[40] in titanic_sruvival_prediction.ipynb)|
|model_selection.png|An image display the accuracy scores of each trained model|
|optimal_model.png|An image display the optimal parameters of Random Forest Classifier|
|rand_rf_model.pkl|The trained Random Forest Classifier (refer to line[61] in titanic_survival_prediction.ipynb)|
|submission.png|An image display the score and ranking of this competition|
|titanic_survival_prediction.ipynb|The main Jupyter Notebook coding file which included all the steps for this competition|
|titanic_test_data.csv|The testing dataset provided in Kaggle|
|titanic_train_data.csv|The training dataset provided in Kaggle|


### THE MODEL SELECTION
The following figure show the accuracy scores of each trained model.
The results shown that the Random Forest model is outperformed the others, it is therefore being selected as the best classifier of this set of data.

![Image](model_selection.png)


### THE OPTIMAL MODEL
The following figure show the optimal parameters of Random Forest Classifier through random search hyperparameter tuning.

![Image](optimal_model.png)


### THE COMPETITION RESULT
The following figure show the ranking and score of my submission.

![Image](submission.png)


### ADDITIONAL
Besides building the machine learning model for this competition, an App was also created to deploy this trained model.
Please refer to "titanic-app-streamlit-heroku" repository. 
[Link to repository](https://github.com/liangchua/titanic-app-streamlit-heroku)

Thank you for taking the time to read my work !!! 
