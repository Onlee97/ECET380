import tensorflow as tf
import tensorflow_datasets as tfds 
datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)
train_size = info.splits["train"].num_examples
batch_size = 32
train_set = datasets["train"].repeat().batch(batch_size).prefetch(1)
train_size = 5000

from keras.preprocessing.text import Tokenizer
# silly example text, you will want to use much more
samples=['The professor is annoyed.','At least this book is better.']
tokenizer=Tokenizer(num_words=20000)
# next command will build the vocabulary,
# keeping only the 20k most frequent words
tokenizer.fit_on_texts(samples)
# now that the tokenizer is fit, map words to ints
sequences=tokenizer.texts_to_sequences(samples)
# the word index that was computed
word_index=tokenizer.word_index

# parse the glove embeddings (at https://nlp.stanford.edu/projects/glove)
embeddings_index={}
f=open('glove.6B.100d.txt')
for line in f:
	values=line.split()
	word=values[0]
	coefs=np.asarray(values[1:],dtype='float32')
	embeddings_index[word]=coefs
f.close()
embedding_dim=100
embedding_matrix=np.zeros((max_words,embedding_dim))
for word,i in word_index.items():
	if i< max_words:
		embedding_vector=embeddings_index.get(word)
		if embedding_vector is not None:
			embeddingMatrix[i]=embedding_vector

model=keras.layers.Sequential([
	keras.layers.Embedding(max_words,embedding_dim,input_length=maxLen),
	keras.layers.GRU(32),
	keras.layers.Dense(1,activation='sigmoid')])
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable=False

history = model.fit(train_set, steps_per_epoch=train_size // batch_size, epochs=5)
