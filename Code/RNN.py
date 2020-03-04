import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

def generate_time_series(batch_size, n_steps):
	freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
	time = np.linspace(0, 1, n_steps)
	series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10)) # wave 1
	series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + wave 2
	series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5) # + noise
	return series[..., np.newaxis].astype(np.float32)

np.random.seed(42)
n_steps = 50
series = generate_time_series(10000, n_steps + 10)
X_train = series[:7000, :n_steps]
X_valid = series[7000:9000, :n_steps]
X_test = series[9000:, :n_steps]
Y = np.empty((10000, n_steps, 10))
for step_ahead in range(1, 10 + 1):
	Y[..., step_ahead - 1] = series[..., step_ahead:step_ahead + n_steps, 0]
Y_train = Y[:7000]
Y_valid = Y[7000:9000]
Y_test = Y[9000:]
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
	keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
	keras.layers.LSTM(20, return_sequences=True),
	keras.layers.TimeDistributed(keras.layers.Dense(10))
	])

model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(X_train, Y_train, epochs=20,
			validation_data=(X_valid, Y_valid))
model.evaluate(X_valid, Y_valid)

# Baseline Metrics Naive predictions (just predict the last observed value):
# y_pred = X_valid[:, -1]
# np.mean(keras.losses.mean_squared_error(y_valid, y_pred))
# # Linear predictions:
# np.random.seed(42)
# tf.random.set_seed(42)

# model = keras.models.Sequential([
# 								keras.layers.Flatten(input_shape=[50, 1]),
# 								keras.layers.Dense(1)
# 								])
# model.compile(loss="mse", optimizer="adam")
# history = model.fit(X_train, y_train, epochs=20,
# 					validation_data=(X_valid, y_valid))

# print(model.evaluate(X_valid, y_valid))
# # A simple RNN
# np.random.seed(42)
# tf.random.set_seed(42)
# model = keras.models.Sequential([
# 	keras.layers.SimpleRNN(1, input_shape=[None, 1])
# 	])

# optimizer = keras.optimizers.Adam(lr=0.005)
# model.compile(loss="mse", optimizer=optimizer)
# history = model.fit(X_train, y_train, epochs=20,
# 					validation_data=(X_valid, y_valid))
# print(model.evaluate(X_valid, y_valid))

