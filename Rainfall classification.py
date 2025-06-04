# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
help(RandomForestClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder


# Step 2: Load dataset
D = pd.read_csv("C:/Users/User/Downloads/Mymensingh (2).csv")
D.head()
D.describe()

print(D.dtypes)

# Step 3: Drop missing values
D.dropna(how='any', inplace=True)
print(D['WID'].unique())
print(D['RAN'].unique())
le=LabelEncoder()
D['WID_ENCODER']=le.fit_transform(D['WID'])
print(le.classes_)
print(le.transform(le.classes_))
print(dict(zip(le.classes_, le.transform(le.classes_))))

# Step 4: Drop unnecessary columns
# ID, Station, Year, Month = identifiers
# T_RAN, A_RAIN = part of RAN info


# Step 5: Separate features and target
X = D.drop(['RAN','WID'], axis=1)
Y = D['RAN']

# Step 6: Normalize feature values (Min-Max scaling)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.25, random_state=31)

# Step 8: Initialize and train MLPClassifier
mlp = MLPClassifier(
    solver='lbfgs',
    alpha=1e-5,
    max_iter=100000,
    activation='logistic',
    hidden_layer_sizes=(4, 3),
    random_state=1,
    verbose=True
)

from sklearn.model_selection import GridSearchCV
from sklearn import svm

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params': {
            'C': [1, 10, 20],
            'kernel': ['rbf', 'linear']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [1, 5, 10]
        }
    },
    'mlp_classifier':{
        'model':MLPClassifier(),
        'params':{
            'activation':['relu','logistic'],
            'solver':['adam','lbfgs'],
            'alpha': [1e-5, 1e-4, 1e-3],
            'max_iter': [1000]}
        }
    
}
Scores=[]
for model_name,mp in model_params.items():
    clf=GridSearchCV(mp['model'],mp['params'],cv=5,return_train_score=False)
clf.fit(X_train,Y_train)
Scores.append({
    'model':model_name,
    'best_score':clf.best_score_,
    'best_params':clf.best_params_,
    })
df=pd.DataFrame(Scores,columns=['model','best_score','best_params'])
    
