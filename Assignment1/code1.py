import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import clone
import numpy as np

#Prepare the data
data = pd.read_pickle('appml-assignment1-dataset.pkl')
y = data['y']
X = data['X'].drop("date", axis = 1)
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size = 0.4)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)


# Normalize the data
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
scaler = Pipeline([
				    ('imputer', SimpleImputer(strategy="median")),
					("std_scaler", StandardScaler()),
				])
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

#Dump pipeline
pickle.dump(scaler, open('pipeline1.pkl', 'wb'))


# Build Model
import tensorflow as tf 
from tensorflow import keras
model = keras.models.Sequential([
	keras.layers.Dense(30, activation="relu", input_shape=X_train_scaled.shape[1:]),
	keras.layers.Dense(10),
	keras.layers.Dropout(0.2),
	keras.layers.Dense(1)
	])

model.compile(loss="mean_squared_error", optimizer="sgd")

#Add Early stopping and call back
checkpoint_cb = keras.callbacks.ModelCheckpoint("model1.h5", save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
history = model.fit(X_train_scaled, y_train, epochs=100,
					validation_data= (X_valid_scaled, y_valid),
					callbacks=[early_stopping_cb, checkpoint_cb]
					)
mse_test = model.evaluate(X_test_scaled, y_test)

# Fit model
y_pred = model.predict(X_test_scaled)
from sklearn.metrics import mean_squared_error
print("mean_squared_error: ",mean_squared_error(y_test, y_pred))
