import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)

def preprocess(X_batch, y_batch):
	X_batch = tf.strings.substr(X_batch, 0, 300)
	X_batch = tf.strings.regex_replace(X_batch, rb"<br\s*/?>", b" ")
	X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z']", b" ")
	X_batch = tf.strings.split(X_batch)
	return X_batch.to_tensor(default_value=b"<pad>"), y_batch

from collections import Counter
vocabulary = Counter()
for X_batch, y_batch in datasets["train"].batch(32).map(preprocess):
	for review in X_batch:
		vocabulary.update(list(review.numpy()))
vocab_size = 10000
truncated_vocabulary = [word for word, count in vocabulary.most_common()[:vocab_size]]
words = tf.constant(truncated_vocabulary)
word_ids = tf.range(len(truncated_vocabulary), dtype=tf.int64)
vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
num_oov_buckets = 1000
table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)

#MODEL
embed_size = 128
model = keras.models.Sequential([
keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size,
	input_shape=[None]),
	keras.layers.GRU(128, return_sequences=True),
	keras.layers.GRU(128),
	keras.layers.Dense(1, activation="sigmoid")
	])
model.compile(loss="binary_crossentropy", optimizer="adam",
metrics=["accuracy"])

#TRAIN
def encode_words(X_batch, y_batch):
	return table.lookup(X_batch), y_batch

train_size = 25000
train_set = datasets["train"].repeat().batch(32).map(preprocess)
train_set = train_set.map(encode_words).prefetch(1)
history = model.fit(train_set, steps_per_epoch=train_size // 32, epochs=5)

# train_size = 32
# test_set = datasets["test"].repeat().batch(32).map(preprocess)
# test_set = test_set.map(encode_words).prefetch(1)
# history = model.fit(train_set, steps_per_epoch=train_size, epochs=5)
