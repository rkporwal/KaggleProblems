import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# importing dataset
dataset=pd.read_csv("titanic_train.csv")

dataset.head()
dataset.info()
dataset.isnull()
dataset.isnull().sum()

sns.headmap()

# Let’s do some Exploratory Data Analysis
sns.heatmap(dataset.isnull(),yticklabels=False,cbar='viridis')

# let’s see what is the ratio of survived v/s not survived.
#people who survived v/s who didn't

sns.set_style('whitegrid')
sns.countplot(x='Survived', data=dataset, palette='RdBu_r')

sns.countplot(x='Survived', hue='Sex', data= dataset,palette='RdBu_r')

sns.countplot(x='Survived', hue='Pclass', data= dataset, palette='rainbow')

sns.distplot(dataset['Age'].dropna(),color='darkred',bins=30)


dataset['Fare'].hist(color='green',bins=40,figsize=(8,4))

# Data Cleaning
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=dataset,palette='winter')


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
            return Age


dataset['Age'] = dataset[['Age', 'Pclass']].apply(impute_age, axis = 1)   

def impute_cabin(col):
  Cabin = col[0]
  if type(Cabin) == str:
    return 1
  else:
      return 0
   
dataset['Cabin'] = dataset[['Cabin']].apply(impute_cabin, axis = 1)


sex = pd.get_dummies(dataset['Sex'],drop_first=True)
embark = pd.get_dummies(dataset['Embarked'],drop_first=True)
dataset['Embarked'].unique()
dataset.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
dataset = pd.concat([dataset,sex,embark],axis=1)
dataset.head()


X_train, X_test, y_train, y_test = train_test_split(dataset.drop('Survived',axis=1),dataset['Survived'], test_size=0.25,random_state=101)

from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(X_train, y_train)
pred = regressor.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pred))
