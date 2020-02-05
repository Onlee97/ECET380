import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np


#Prepare the data
data = pd.read_pickle('appml-assignment1-dataset.pkl')
y = data['y']
X = data['X'][['CAD-open', 'CAD-high', 'CAD-low', 'CAD-close']]
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size = 0.3)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

print(X_train)

# prepare the data
poly_scaler = Pipeline([("std_scaler", StandardScaler())])


X_train_poly_scaled = poly_scaler.fit_transform(X_train)
X_val_poly_scaled = poly_scaler.transform(X_valid)

from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True,
						penalty=None, learning_rate="constant", eta0=0.0005)
						# warm_start=True means when fit is called again, continues training where left off
minimum_val_error = float("inf")
best_epoch = None
best_model = None
for epoch in range(10):
	sgd_reg.fit(X_train_poly_scaled, y_train) # continues where it left off
	y_val_predict = sgd_reg.predict(X_val_poly_scaled)
	val_error = mean_squared_error(y_valid, y_val_predict)
	print("mean_squared_error: ", val_error)
	if val_error < minimum_val_error:
		minimum_val_error = val_error
		best_epoch = epoch
		best_model = clone(sgd_reg)