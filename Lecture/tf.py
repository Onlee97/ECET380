import tensorflow as tf 
from tensorflow import keras

print(tf.__version__)
print(keras.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()


X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] /255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
				"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

model = keras.models.Sequential([
	keras.layers.Flatten(input_shape=[28,28]),
	keras.layers.Dense(300, activation="relu"),
	keras.layers.Dense(100, activation="relu"),
	keras.layers.Dense(10, activation="softmax"),
	])

model.summary()


model.compile(loss="sparse_categorical_crossentropy",
			optimizer="sgd",
			metrics=["accuracy"]
			)

history = model.fit(X_train, y_train, epochs=30,
					validation_data= (X_valid, y_valid))

import pandas as pd
import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

model.evaluate(X_test, y_test)