from keras import backend as K

class Sampling(keras.layers.Layers):
	def call(self, inputs):
		mean, log_var = inputs
		return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean


tf.random.set_seed(42)
np.random.seed(42)
codings_size = 10
inputs = keras.layers.Input(shape=[28, 28])

z = keras.layers.Flatten()(inputs)
z = keras.layers.Dense(150, activation="selu")(z)
z = keras.layers.Dense(100, activation="selu")(z)
codings_mean = keras.layers.Dense(codings_size)(z)
codings_log_var = keras.layers.Dense(codings_size)(z)
codings = Sampling()([codings_mean, codings_log_var])
variational_encoder = keras.models.Model(
inputs=[inputs], outputs=[codings_mean, codings_log_var, codings])

decoder_inputs = keras.layers.Input(shape=[codings_size])
x = keras.layers.Dense(100, activation="selu")(decoder_inputs)
x = keras.layers.Dense(150, activation="selu")(x)
x = keras.layers.Dense(28 * 28, activation="sigmoid")(x)