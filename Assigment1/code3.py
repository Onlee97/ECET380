import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
import numpy as np

#Prepare the data
data = pd.read_pickle('appml-assignment1-dataset.pkl')
y = data['y']
X = data['X'].drop("date", axis = 1)
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size = 0.4)


# Normalize the data
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
scaler = Pipeline([
				    ('imputer', SimpleImputer(strategy="median")),
					("std_scaler", StandardScaler()),
				])
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)

#Dump pipeline
pickle.dump(scaler, open('pipeline3.pkl', 'wb'))

# Fit model
from sklearn.linear_model import ElasticNet
model = ElasticNet(alpha = 0.1, l1_ratio = 0.1)
model.fit(X_train_scaled, y_train_full)
y_pred = model.predict(X_test_scaled)
print("mean_squared_error: ",mean_squared_error(y_test, y_pred))

#Dump model
pickle.dump(model, open('model2.pkl', 'wb'))
