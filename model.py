import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import re

df = pd.read_csv('s.csv')

df['ETA']= pd.to_datetime(df['ETA'],format="%d/%m/%Y %H:%M",errors='coerce')
df['year'] = pd.DatetimeIndex(df['ETA']).year
df['month'] = pd.DatetimeIndex(df['ETA']).month
df['day'] = pd.DatetimeIndex(df['ETA']).day
#df=df.drop(df['ETA'], axis=1)
#k=re.split('/ |, ',j )
df= df[['Nom du port','Nom du Navire','day','month','year','Import']]
df['day'].fillna(df['day'].mean(), inplace=True)
df['month'].fillna(df['month'].mean(), inplace=True)
df['year'].fillna(df['year'].mean(), inplace=True)
a=df.filter(regex='(Nom|day|month|year|Import)')

# Create the dataframe df
X=a.iloc[:,0:5]
y=a.iloc[:,5]



categorical_feature_mask = X.dtypes==object
categorical_cols = X.columns[categorical_feature_mask].tolist()
from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder()
le2 = LabelEncoder()
#le3 = LabelEncoder()
X['Nom du port'] = le1.fit_transform(X['Nom du port']) 

X['Nom du Navire'] = le2.fit_transform(X['Nom du Navire']) 


#X['ETA'] = le3.fit_transform(X['ETA']) 

#***********************************************
np.save('classes1.npy',le1.classes_)
np.save('classes2.npy',le2.classes_)
#np.save('classes3.npy',le3.classes_)


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  
from sklearn.naive_bayes import GaussianNB

gnb=GaussianNB()
from sklearn.model_selection import train_test_split
model=gnb.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn import metrics
c2=metrics.accuracy_score(y_test, y_pred)

from sklearn.metrics import f1_score
f2=f1_score(y_test, y_pred, average='macro')
print("GaussianNB: " )
print("accuracy est égal : " )
print(c2)
print("f1_score est égal : " )
print(f2)

# Saving model to disk
pickle.dump(gnb, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[1,2,20/11/2020]]))
print(X)
print(le1.classes_)
print(le2.classes_)
