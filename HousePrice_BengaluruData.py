import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# importing dataset
dataset=pd.read_csv("Bengaluru_House_Data.csv")

dataset.shape
dataset.head()

dataset.groupby(['area_type'])['area_type'].agg('count')

df2=dataset.drop(['area_type','society','balcony','availability'],axis='columns')
dataset.columns
df2.head()

df2.isnull().sum()
df2['size'].unique()

df3 = df2.dropna()

df3['bhk']=df3['size'].apply(lambda x: int(x.split(' ')[0]))

df3['bhk'].unique()
df3[df3.bhk>20]
df3.total_sqft.unique()

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

df3[~df3['total_sqft'].apply(is_float)].head(10)


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
    
convert_sqft_to_num('22')    

convert_sqft_to_num('22-44')  


convert_sqft_to_num('34.46Sq. Meter')  

df4=df3.copy()
df4['total_sqft']=df4['total_sqft'].apply(convert_sqft_to_num )
df4.head()

df4.loc[30]

  
