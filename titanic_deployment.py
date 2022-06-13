# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 13:37:12 2022

@author: USER
"""

# Import libraries/packages
import pandas as pd
import pickle 
import os
import streamlit as st
from PIL import Image

# Call the saved models path
MINMAX_PATH = os.path.join(os.getcwd(),'minmax.pkl')
MODEL_PATH = os.path.join(os.getcwd(),'rand_rf_model.pkl')

# Load the saved models for deployment
minmax = pickle.load(open(MINMAX_PATH,'rb'))
model = pickle.load(open(MODEL_PATH,'rb'))

# Header in app
st.write("""
         # The Titanic Survival Prediction
        
         This app predicts the survival of Titanic's passengers!
         """)

# Call an image
im = Image.open(r"Titanic_2.jpg")

# Load the image to the app 
st.image(im, width=700, caption='The Titanic')

# Add subheader in sidebar         
st.sidebar.header('Input Parameters')
st.sidebar.write("""Please refer to the tables at right for the description of each parameter""")

# Create a function to display the parameters and ask for user input    
def input_features():
    pclass = st.sidebar.selectbox('Ticket Class', (1, 2, 3))
    sex = st.sidebar.selectbox('Gender', ('Male','Female'))
    embarked = st.sidebar.selectbox('Embarked from', ('Cherbourg','Queenstown','Southampton'))
    age = st.sidebar.slider('Age Group', 0, 4, 2)
    sibsp = st.sidebar.slider('Number of Sibilings/Spouses Aboard Together', 0, 8, 2)
    parch = st.sidebar.slider('Number of Parents/Children Aboard Together', 0, 9, 3)
    fare = st.sidebar.slider('Passenger Fare Group', 0, 4, 3)
    title = st.sidebar.slider('Passenger Title', 0, 5, 1)
    
    data = {'Pclass': pclass,
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Fare': fare,
            'Title': title,
            'Sex': sex,
            'Embarked': embarked}
    
    features = pd.DataFrame(data, index=[0])
    return features

# Store the user input data to df
df = input_features()

# Encode the sex data (to fit the trained model)
def encode_sex(data):
    data.loc[data['Sex'] == 'Female', 'Sex_female'] = 1
    data.loc[data['Sex'] == 'Male', 'Sex_female'] = 0
    data.loc[data['Sex'] == 'Male', 'Sex_male'] = 1
    data.loc[data['Sex'] == 'Female', 'Sex_male'] = 0
    data['Sex_female'] = data['Sex_female'].astype(int)
    data['Sex_male'] = data['Sex_male'].astype(int)

# Encode the embarked data (to fit the trained model)    
def encode_embarked(data):
    data.loc[data['Embarked'] == 'Cherbourg', 'Embarked_C'] = 1
    data.loc[data['Embarked'] == 'Queenstown', 'Embarked_C'] = 0
    data.loc[data['Embarked'] == 'Southampton', 'Embarked_C'] = 0
    data.loc[data['Embarked'] == 'Cherbourg', 'Embarked_Q'] = 0
    data.loc[data['Embarked'] == 'Queenstown', 'Embarked_Q'] = 1
    data.loc[data['Embarked'] == 'Southampton', 'Embarked_Q'] = 0
    data.loc[data['Embarked'] == 'Cherbourg', 'Embarked_S'] = 0
    data.loc[data['Embarked'] == 'Queenstown', 'Embarked_S'] = 0
    data.loc[data['Embarked'] == 'Southampton', 'Embarked_S'] = 1
    data['Embarked_C'] = data['Embarked_C'].astype(int)
    data['Embarked_Q'] = data['Embarked_Q'].astype(int)
    data['Embarked_S'] = data['Embarked_S'].astype(int)

# Add not_alone feature (to fit the trained model)
def add_notalone(data):
    data['Relatives'] = data['SibSp'] + data['Parch']

    data.loc[data['Relatives'] > 0, 'Not_Alone'] = 0
    data.loc[data['Relatives'] == 0, 'Not_Alone'] = 1

# Add fare_per_person feature (to fit the trained model)
def add_fareperson(data):
    data['Fare_Per_Person'] = data['Fare']/(data['Relatives']+1)
    data['Fare_Per_Person'] = data['Fare_Per_Person'].astype(int)

# Add age_class feature (to fit the trained model)
def add_ageclass(data):
    data['Age_Class'] = data['Age']*data['Pclass']

# Transform all data into 0 - 1 (to fit the trained model)
def min_max(data):
    to_mm = ['Pclass','Age','SibSp','Parch','Fare','Title','Relatives',
             'Fare_Per_Person','Age_Class']
    data[to_mm] = minmax.transform(data[to_mm])

# Clean/Transform the input data
encode_sex(df)
encode_embarked(df)
df = df.drop(['Sex','Embarked'], axis=1)

# Display the user input data in app
st.subheader('User Input Parameters')
st.write("""You may change the parameters in the left side bar to predict
        the survival of the Titanic's passenger.""")
st.write(df)

# Clean/Transform the user input data
add_notalone(df)
add_fareperson(df)
add_ageclass(df)
min_max(df)

# Deploy the trained model and get the prediction result
result = model.predict(df)
output = pd.DataFrame({'Survived':result}, index=['Prediction Result'])

# Display the output
st.subheader('Prediction')

pred1, pred2, pred3, pred4 = st.columns(4)
pred1.write("""The predicted result: """)
pred2.write("0 = Did Not Survive")
pred3.write("1 = Survived")
pred4.text("")
st.write(output)

# Display the description of parameters 
st.subheader('Description of Parameters')
st.text('')
with st.expander("Click and Open to see *Features Description*"):
    st.write("""
         Pclass : Class of ticket purchased
         
         Age : Age of passenger 
         
         SibSp : Number of siblings and/or spouses aboard together
         
         Parch : Number of parents and/or children aboard together
         
         Fare : Fare of ticket
         
         Title : Passenger's title
         
         Sex_female : Gender of passenger is female 
         
         Sex_male : Gender of passenger is male
         
         Embarked_C : Passenger is embarked from Cherbourg
         
         Embarked_Q : Passenger is embarked from Queenstown
         
         Embarked_S : Passenger is embarked from Southampton
         """)
    st.text('')
    st.text('')
st.write("""The following tables shown the data range of some parameters
         for your reference.""")

with st.expander("Click and Open to see the *Tables*"):         
    col1, col2 = st.columns(2) 

    col1.write("""
         Ticket Class:
         |Ticket Class|Label|
         |--------------|:----------:|
         |1st class (upper)|1|
         |2nd class (middle)|2|
         |3rd class (lower)|3|
         """) 
      
    col2.write("""             
         Age Group:
         |Age Range |Label|
         |:--------:|:-------:|
         |< 19      |0        |
         |19 - 24     |1        |
         |25 - 31     |2        |
         |32 - 39     |3        |
         |>= 40     |4        |
         """)
    st.text('')

    col3, col4 = st.columns(2) 

    col3.write("""             
         Fare Group:
         |Fare Range |Label|
         |:--------:|:-------:|
         |< 7.85      |0        |
         |7.85 - 10.49  |1        |
         |10.50 - 21.67 |2        |
         |21.68 - 39.68 |3        |
         |>= 39.69    |4        |
         """)

    col4.write("""
          Passenger's Title:
          |Title |Label|
          |:--------:|:-------:|
          |Not Available|0     |
          |Mr.          |1     |
          |Miss.        |2     |
          |Mrs.         |3     |
          |Master       |4     |
          |Other        |5     |
           """)
    st.text('')
   