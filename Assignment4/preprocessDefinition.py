import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import tensorflow_addons as tfa
import seaborn as sns
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def embed(input):
	return model(input)

def plot_similarity(labels, features, rotation):
	corr = np.inner(features, features)
	sns.set(font_scale=1.2)
	g = sns.heatmap(corr,xticklabels=labels,
					yticklabels=labels,vmin=0,vmax=1,
					cmap="YlOrRd")
	g.set_xticklabels(labels, rotation=rotation)
	g.set_title("Semantic Textual Similarity")

def run_and_plot(messages_):
	message_embeddings_ = embed(messages_)
	plot_similarity(messages_, message_embeddings_, 90)

def preprocess(data_point):
	X_batch = data_point['text']
	y_batch = data_point['title']
	X_batch = tf.strings.substr(X_batch, 0, 300)
	X_batch = tf.strings.regex_replace(X_batch, b"<br\\s*/?>", b" ")
	X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z']", b" ")
	X_batch = tf.strings.split(X_batch)
	return X_batch.to_tensor(default_value=b"<pad>"), y_batch

if __name__ == "__main__":
	module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
	model = hub.load(module_url)
	
	sentences = ["I am a student.", "A student is me"]
	run_and_plot(sentences)
	plt.show()
