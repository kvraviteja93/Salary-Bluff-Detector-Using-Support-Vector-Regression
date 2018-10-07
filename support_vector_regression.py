#Importing The Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing The Dataset
dataset=pd.read_csv("Position_Salaries.csv")

#Separating Independent and Dependent Variables
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()
X=sc_X.fit_transform(X)
y=np.squeeze(sc_y.fit_transform(y.reshape(-1,1)))

#Fitting Support Vector Regression To The Dataset
from sklearn.svm import SVR
svr=SVR(kernel='rbf')
svr.fit(X,y)

#Predicting The Results
y_pred=sc_y.inverse_transform(svr.predict(sc_X.transform(np.array([[6.5]]))))

#Visualizing The Support Vector Regression Results
plt.scatter(X,y,color='red')
plt.plot(X,svr.predict(X),color='green')
plt.title("Salary Bluff Detector Using SVR")
plt.xlabel("Positions")
plt.ylabel("Salaries")
plt.show()