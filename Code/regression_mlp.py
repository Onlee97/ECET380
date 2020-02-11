from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf 
from tensorflow import keras

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.fit_transform(X_valid)
X_test_scaled = scaler.fit_transform(X_test)

model = keras.models.Sequential([
	keras.layers.Dense(30, activation="relu", input_shape=X_train_scaled.shape[1:]),
	keras.layers.Dense(1)
	])

model.compile(loss="mean_squared_error", optimizer="sgd")

checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)
history = model.fit(X_train_scaled, y_train, epochs=20,
					validation_data= (X_valid_scaled, y_valid),
					callbacks=[checkpoint_cb]
					)

model = keras.models.load_model("my_keras_model.h5")

early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

history = model.fit(X_train_scaled, y_train, epochs=100,
					validation_data= (X_valid_scaled, y_valid),
					callbacks=[checkpoint_cb, early_stopping_cb]
					)

mse_test = model.evaluate(X_test_scaled, y_test)
X_new = X_test_scaled[:3]
y_pred = model.predict(X_new)

