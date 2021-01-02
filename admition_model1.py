# checkout https://assig6-icfoss.herokuapp.com/
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

pickle_in = open('admition_model_regre.pkl', 'rb')
classifier = pickle.load(pickle_in)

st.sidebar.header('The prediction of Graduate Admissions from an Indian perspective using regression model')

"""
### The prediction of Graduate Admissions from an Indian perspective using regression model.

The dataset contains several parameters which are considered important during the application for Masters Programs.
 
The parameters included are :
> inputs

1.	GRE Scores ( out of 340 )
2.	TOEFL Scores ( out of 120 )
3.	University Rating ( out of 5 )
4.	Statement of Purpose  ( out of 5 )
5.  Letter of Recommendation Strength ( out of 5 )
6.	Undergraduate CGPA ( out of 10 )
7.	Research Experience ( either 0 or 1 )

> Output/prediction

1.	Chance of Admition ( ranging from 0 to 1 )"""

gre = st.slider('GRE Score', min_value=0, max_value=340, step=5, value=100, key='gre')
toefl = st.slider('TOEFL Score', min_value=0, max_value=120, step=2, value=50, key='toefl')
univ_rating = st.slider('University Rating', min_value=1.0, max_value=5.0, step=0.5, value=2.0, key='univ')
sop = st.slider('Statement of Purpose strength', min_value=1.0, max_value=5.0, step=0.5, value=2.0, key='sop')
lor = st.slider('Letter of recommendation strength', min_value=1.0, max_value=5.0, step=0.5, value=2.0, key='lor')
cgpa = st.slider('Undergraduate CGPA', min_value=1.0, max_value=10.0, step=0.25, value=5.0, key='gpa')
research = st.select_slider('Research Experience ( either 0 or 1 )', options=(0,1))

submit = st.button('Predict admission chance')
if submit:
    prediction = classifier.predict([[gre, toefl, univ_rating, sop, lor, cgpa, research]])
    st.write(f'Chance of admission is {prediction[0]*100:.2f}%')
