import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataframe = pd.read_csv("Employee-Attrition.csv")

dataframe.info()
dataframe.describe()

# Exploratory data analysis 

dataframe['Attrition'].unique()
dataframe['Attrition'].value_counts()

categorical_features=[]
for col in dataframe.columns:
    if(dataframe[col].dtype=='object'):
        categorical_features.append(col)
        
for col in categorical_features:
            print(col,":",len(dataframe[col].unique()))
            

plt.scatter(dataframe['Age'],dataframe['Attrition'])          
          
plt.scatter(dataframe['DailyRate'],dataframe['Attrition']) 
plt.scatter(dataframe['DistanceFromHome'],dataframe['Attrition']) 
plt.scatter(dataframe['Education'],dataframe['Attrition']) 
plt.scatter(dataframe['EnvironmentSatisfaction'],dataframe['Attrition']) 
plt.scatter(dataframe['JobSatisfaction'],dataframe['Attrition']) 
plt.scatter(dataframe['YearsAtCompany'],dataframe['Attrition']) 

# features importance, 
correlation_matrix=dataframe.corr()

