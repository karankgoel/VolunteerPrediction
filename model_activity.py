import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix,auc,roc_auc_score
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score

from sklearn.model_selection import GridSearchCV
import pickle

df_original = pd.read_csv("Book1.csv", na_values=' ')

df = df_original.copy()

df.drop(columns=['id', 'previous activity type1'], inplace=True)

df['previous'] = df['previous activity type']

df['volunteered_before'] = df['volunteered before']

df.drop(['previous activity type','volunteered before'], axis=1, inplace=True)

df.age.fillna(df.age.mean(), inplace=True)

df.gender.fillna(0, inplace=True)

df.hobbies_sports.fillna(0, inplace=True)

df.hobbies_environment.fillna(0, inplace=True)

df.hobbies_fitness.fillna(0, inplace=True)

df.hobbies_cooking.fillna(0, inplace=True)

df['volunteered_before'] = pd.Categorical(df['volunteered_before']).codes

df['differently abled'] = pd.Categorical(df['differently abled']).codes

Y_volunteered = df['volunteered_before']

Y_activity = df['previous']

X = df.drop(['volunteered_before', 'previous'], axis = 1)

for col in ['age', 'gender', 'hobbies_sports', 'hobbies_environment','hobbies_cooking','hobbies_fitness']:
  df[col] = df[col].astype('int64')

previous_attended = df[df['volunteered_before'] == 1]

previous_attended.drop('volunteered_before', axis = 1, inplace=True)

previous_attended['previous'] = previous_attended['previous'].map({'on-field': 0, 'spot': 1, 'social': 3, 'moving': 2})

Y_previous = previous_attended['previous']

X_previous = previous_attended.drop('previous', axis=1)

X_previous_train, X_previous_test, y_previous_train, y_previous_test = train_test_split(X_previous, Y_previous, test_size=0.3,random_state = 1)

nb_model = GaussianNB()

nb_model.fit(X_previous_train, y_previous_train)

pickle.dump(nb_model, open('model_activity.pkl','wb'))

# Loading model to compare the results
model_activity = pickle.load(open('model_activity.pkl','rb'))


