import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import tensorflow_addons as tfa
import seaborn as sns
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessDefinition import preprocess
from collections import Counter


def loadWikiDataset():
	dataset, info = tfds.load("wikipedia/20190301.en", split='train[:1%]' ,with_info=True)
	train_set = tfds.load("wikipedia/20190301.en", split='train[:1%]')
	valid_set = tfds.load("wikipedia/20190301.en", split='train[2%:3%]')

	train_size = info.splits['train[:1%]'].num_examples
	valid_size = info.splits['train[2%:3%]'].num_examples

	print(train_size, valid_size)

	for X_batch, y_batch in train_set.take(1).batch(32).map(preprocess):
		print(X_batch)

	# train_set.take(1).splits("test")
	# vocabulary = Counter()
	# for X_batch, y_batch in train_set.take(1000).batch(32).map(preprocess):
	# 	for review in X_batch:
	# 		vocabulary.update(list(review.numpy()))

	# vocab_size = 10000
	# truncated_vocabulary = [word for word, count in vocabulary.most_common()[:vocab_size]]

	# print(vocabulary.most_common()[:3])

	# words = tf.constant(truncated_vocabulary)
	# word_ids = tf.range(len(truncated_vocabulary), dtype=tf.int64)
	# vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
	# num_oov_buckets = 1000
	# table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)

	# tokenize = table.lookup(tf.constant([b"This movie was nice".split()]))

	# print(tokenize)

	
	# module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
	# encoder = hub.load(module_url)

	# encoder_inputs = keras.layers.Input(shape=[None], dtype=np.int32)
	# decoder_inputs = keras.layers.Input(shape=[None], dtype=np.int32)
	# sequence_lengths = keras.layers.Input(shape=[], dtype=np.int32)

	# embeddings = keras.layers.Embedding(vocab_size, embed_size)
	# encoder_embeddings = embeddings(encoder_inputs)
	# decoder_embeddings = embeddings(decoder_inputs)

	# encoder = keras.layers.LSTM(512, return_state=True)
	# encoder_outputs, state_h, state_c = encoder(encoder_embeddings)
	# encoder_state = [state_h, state_c]

	# sampler = tfa.seq2seq.sampler.TrainingSampler()

	# decoder_cell = keras.layers.LSTMCell(512)
	# output_layer = keras.layers.Dense(vocab_size)
	# decoder = tfa.seq2seq.basic_decoder.BasicDecoder(decoder_cell, sampler,
	#                                                  output_layer=output_layer)
	# final_outputs, final_state, final_sequence_lengths = decoder(
	#     decoder_embeddings, initial_state=encoder_state,
	#     sequence_length=sequence_lengths)
	# Y_proba = tf.nn.softmax(final_outputs.rnn_output)

	# model = keras.Model(inputs=[encoder_inputs, decoder_inputs, sequence_lengths],
	#                     outputs=[Y_proba])

def encode_words(X_batch, y_batch):
    return table.lookup(X_batch), y_batch




if __name__ == "__main__":

	loadWikiDataset()



