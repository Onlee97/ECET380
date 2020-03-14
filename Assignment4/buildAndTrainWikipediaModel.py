import csv
import string
import re
from pickle import dump
from unicodedata import normalize
from numpy import array
import itertools
from pickle import load
from tensorflow.keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from pickle import load
from numpy import array
from numpy import argmax

# import tensorflow as tf
import tensorflow.compat.v2 as tf
from keras.models import load_model
from sklearn.model_selection import train_test_split
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import numpy as np

def loadWikiDataset():
	dataset, info = tfds.load("wikipedia/20190301.en", split='train[:1%]' ,with_info=True)
	train_set = tfds.load("wikipedia/20190301.en", split='train[:1%]')
	valid_set = tfds.load("wikipedia/20190301.en", split='train[2%:3%]')

	train_size = info.splits['train[:1%]'].num_examples
	valid_size = info.splits['train[2%:3%]'].num_examples

	print("train_size: ",train_size)
	print("valid_size: ",valid_size)

	return train_set, valid_set

def preprocess(data_point):
	X_batch = data_point['text']
	X_batch = tf.strings.substr(X_batch, 0, 300)
	X_batch = tf.strings.regex_replace(X_batch, b"<br\\s*/?>", b" ")
	X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z']", b" ")
	X_batch = tf.strings.split(X_batch)

	y_batch = data_point['title']
	y_batch = tf.strings.regex_replace(y_batch, b"<br\\s*/?>", b" ")
	y_batch = tf.strings.regex_replace(y_batch, b"[^a-zA-Z']", b" ")
	y_batch = tf.strings.split(y_batch)
	return X_batch.to_tensor(default_value=b"<pad>"), y_batch.to_tensor(default_value=b"<pad>")


rawTextData = list()
rawTitleData = list()
train_set, valid_set = loadWikiDataset()
for X, y in train_set.batch(1).map(preprocess):
	rawTextData.append(str(X))
	rawTitleData.append(str(y))

raw_data_en = list()
raw_data_ge = list()
# raw_data = list()
for X, y in train_set.batch(1).map(preprocess):
	raw_data_en.append(str(X))
	raw_data_ge.append(str(y))
	# raw_data.append(str(X))
	# raw_data.append(str(y))

en_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
en_tokenizer.fit_on_texts(raw_data_en)

data_en = en_tokenizer.texts_to_sequences(raw_data_en)
data_en = tf.keras.preprocessing.sequence.pad_sequences(data_en,padding='post')

ge_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
ge_tokenizer.fit_on_texts(raw_data_ge)

data_ge = ge_tokenizer.texts_to_sequences(raw_data_ge)
data_ge = tf.keras.preprocessing.sequence.pad_sequences(data_ge,padding='post')


# print("Tokenized Text: ", data_en[1])
# print("Raw text: ", raw_data_en[1])
# print("Tokenized title: ", data_ge[1])
# print("Raw title: ",raw_data_ge[1])

def max_len(tensor):
	#print( np.argmax([len(t) for t in tensor]))
	return max( len(t) for t in tensor)

X_train,  X_test, Y_train, Y_test = train_test_split(data_en,data_ge,test_size=0.2)
BATCH_SIZE = 64
BUFFER_SIZE = len(X_train)
steps_per_epoch = BUFFER_SIZE//BATCH_SIZE
embedding_dims = 256
rnn_units = 1024
dense_units = 1024
Dtype = tf.float32   #used to initialize DecoderCell Zero state

Tx = max_len(data_en)
Ty = max_len(data_ge)  

input_vocab_size = len(en_tokenizer.word_index)+1  
output_vocab_size = len(ge_tokenizer.word_index)+ 1
dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
example_X, example_Y = next(iter(dataset))
#print(example_X.shape) 
#print(example_Y.shape) 

#ENCODER
class EncoderNetwork(tf.keras.Model):
	def __init__(self,input_vocab_size,embedding_dims, rnn_units ):
		super().__init__()
		self.encoder_embedding = tf.keras.layers.Embedding(input_dim=input_vocab_size,
														   output_dim=embedding_dims)
		self.encoder_rnnlayer = tf.keras.layers.LSTM(rnn_units,return_sequences=True, 
													 return_state=True )
	
#DECODER
class DecoderNetwork(tf.keras.Model):
	def __init__(self,output_vocab_size, embedding_dims, rnn_units):
		super().__init__()
		self.decoder_embedding = tf.keras.layers.Embedding(input_dim=output_vocab_size,
														   output_dim=embedding_dims) 
		self.dense_layer = tf.keras.layers.Dense(output_vocab_size)
		self.decoder_rnncell = tf.keras.layers.LSTMCell(rnn_units)
		# Sampler
		self.sampler = tfa.seq2seq.sampler.TrainingSampler()
		# Create attention mechanism with memory = None
		self.attention_mechanism = self.build_attention_mechanism(dense_units,None,BATCH_SIZE*[Tx])
		self.rnn_cell =  self.build_rnn_cell(BATCH_SIZE)
		self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler= self.sampler,
												output_layer=self.dense_layer)

	def build_attention_mechanism(self, units,memory, memory_sequence_length):
		return tfa.seq2seq.LuongAttention(units, memory = memory, 
										  memory_sequence_length=memory_sequence_length)
		#return tfa.seq2seq.BahdanauAttention(units, memory = memory, memory_sequence_length=memory_sequence_length)

	# wrap decodernn cell  
	def build_rnn_cell(self, batch_size ):
		rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnncell, self.attention_mechanism,
												attention_layer_size=dense_units)
		return rnn_cell
	
	def build_decoder_initial_state(self, batch_size, encoder_state,Dtype):
		decoder_initial_state = self.rnn_cell.get_initial_state(batch_size = batch_size, 
																dtype = Dtype)
		decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state) 
		return decoder_initial_state

encoderNetwork = EncoderNetwork(input_vocab_size,embedding_dims, rnn_units)
decoderNetwork = DecoderNetwork(output_vocab_size,embedding_dims, rnn_units)
optimizer = tf.keras.optimizers.Adam()

def loss_function(y_pred, y):
   
	#shape of y [batch_size, ty]
	#shape of y_pred [batch_size, Ty, output_vocab_size] 
	sparsecategoricalcrossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
																				  reduction='none')
	loss = sparsecategoricalcrossentropy(y_true=y, y_pred=y_pred)
	mask = tf.logical_not(tf.math.equal(y,0))   #output 0 for y=0 else output 1
	mask = tf.cast(mask, dtype=loss.dtype)
	loss = mask* loss
	loss = tf.reduce_mean(loss)
	return loss


def train_step(input_batch, output_batch,encoder_initial_cell_state):
	#initialize loss = 0
	loss = 0
	with tf.GradientTape() as tape:
		encoder_emb_inp = encoderNetwork.encoder_embedding(input_batch)
		a, a_tx, c_tx = encoderNetwork.encoder_rnnlayer(encoder_emb_inp, 
														initial_state =encoder_initial_cell_state)

		#[last step activations,last memory_state] of encoder passed as input to decoder Network
		
		 
		# Prepare correct Decoder input & output sequence data
		decoder_input = output_batch[:,:-1] # ignore <end>
		#compare logits with timestepped +1 version of decoder_input
		decoder_output = output_batch[:,1:] #ignore <start>


		# Decoder Embeddings
		decoder_emb_inp = decoderNetwork.decoder_embedding(decoder_input)

		#Setting up decoder memory from encoder output and Zero State for AttentionWrapperState
		decoderNetwork.attention_mechanism.setup_memory(a)
		decoder_initial_state = decoderNetwork.build_decoder_initial_state(BATCH_SIZE,
																		   encoder_state=[a_tx, c_tx],
																		   Dtype=tf.float32)
		
		#BasicDecoderOutput        
		outputs, _, _ = decoderNetwork.decoder(decoder_emb_inp,initial_state=decoder_initial_state,
											   sequence_length=BATCH_SIZE*[Ty-1])

		logits = outputs.rnn_output
		#Calculate loss

		loss = loss_function(logits, decoder_output)

	#Returns the list of all layer variables / weights.
	variables = encoderNetwork.trainable_variables + decoderNetwork.trainable_variables  
	# differentiate loss wrt variables
	gradients = tape.gradient(loss, variables)

	#grads_and_vars – List of(gradient, variable) pairs.
	grads_and_vars = zip(gradients,variables)
	optimizer.apply_gradients(grads_and_vars)
	return loss

#RNN LSTM hidden and memory state initializer
def initialize_initial_state():
		return [tf.zeros((BATCH_SIZE, rnn_units)), tf.zeros((BATCH_SIZE, rnn_units))]

epochs = 5
for i in range(1, epochs+1):

	encoder_initial_cell_state = initialize_initial_state()
	total_loss = 0.0

	for ( batch , (input_batch, output_batch)) in enumerate(dataset.take(steps_per_epoch)):
		batch_loss = train_step(input_batch, output_batch, encoder_initial_cell_state)
		total_loss += batch_loss
		if (batch+1)%5 == 0:
			print("total loss: {} epoch {} batch {} ".format(batch_loss.numpy(), i, batch+1))


#In this section we evaluate our model on a raw_input converted to german, for this the entire sentence has to be passed
#through the length of the model, for this we use greedsampler to run through the decoder
#and the final embedding matrix trained on the data is used to generate embeddings
input_raw='The horse (Equus ferus caballus)[2][3] is one of two extant subspecies of Equus ferus. It is an odd-toed ungulate mammal belonging to the taxonomic family Equidae. The horse has evolved over the past 45 to 55 million years from a small multi-toed creature, Eohippus, into the large, single-toed animal of today. Humans began domesticating horses around 4000 BC, and their domestication is believed to have been widespread by 3000 BC.'

# We have a transcript file containing English-Hindi pairs
# Preprocess X
input_lines = ['<start> '+input_raw+'']
input_sequences = [[en_tokenizer.word_index[w] for w in line.split(' ')] for line in input_lines]
input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences,
                                                                maxlen=Tx, padding='post')
inp = tf.convert_to_tensor(input_sequences)
#print(inp.shape)
inference_batch_size = input_sequences.shape[0]
encoder_initial_cell_state = [tf.zeros((inference_batch_size, rnn_units)),
                              tf.zeros((inference_batch_size, rnn_units))]
encoder_emb_inp = encoderNetwork.encoder_embedding(inp)
a, a_tx, c_tx = encoderNetwork.encoder_rnnlayer(encoder_emb_inp,
                                                initial_state =encoder_initial_cell_state)
print('a_tx :',a_tx.shape)
print('c_tx :', c_tx.shape)

start_tokens = tf.fill([inference_batch_size],ge_tokenizer.word_index['<start>'])

end_token = ge_tokenizer.word_index['<end>']

greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

decoder_input = tf.expand_dims([ge_tokenizer.word_index['<start>']]* inference_batch_size,1)
decoder_emb_inp = decoderNetwork.decoder_embedding(decoder_input)

decoder_instance = tfa.seq2seq.BasicDecoder(cell = decoderNetwork.rnn_cell, sampler = greedy_sampler,
                                            output_layer=decoderNetwork.dense_layer)
decoderNetwork.attention_mechanism.setup_memory(a)
#pass [ last step activations , encoder memory_state ] as input to decoder for LSTM
print("decoder_initial_state = [a_tx, c_tx] :",np.array([a_tx, c_tx]).shape)
decoder_initial_state = decoderNetwork.build_decoder_initial_state(inference_batch_size,
                                                                   encoder_state=[a_tx, c_tx],
                                                                   Dtype=tf.float32)
print("\nCompared to simple encoder-decoder without attention, the decoder_initial_state \
 is an AttentionWrapperState object containing s_prev tensors and context and alignment vector \n ")
print("decoder initial state shape :",np.array(decoder_initial_state).shape)
print("decoder_initial_state tensor \n", decoder_initial_state)

# Since we do not know the target sequence lengths in advance, we use maximum_iterations to limit the translation lengths.
# One heuristic is to decode up to two times the source sentence lengths.
maximum_iterations = tf.round(tf.reduce_max(Tx) * 2)

#initialize inference decoder
decoder_embedding_matrix = decoderNetwork.decoder_embedding.variables[0] 
(first_finished, first_inputs,first_state) = decoder_instance.initialize(decoder_embedding_matrix,
                             start_tokens = start_tokens,
                             end_token=end_token,
                             initial_state = decoder_initial_state)
#print( first_finished.shape)
print("\nfirst_inputs returns the same decoder_input i.e. embedding of  <start> :",first_inputs.shape)
print("start_index_emb_avg ", tf.reduce_sum(tf.reduce_mean(first_inputs, axis=0))) # mean along the batch

inputs = first_inputs
state = first_state  
predictions = np.empty((inference_batch_size,0), dtype = np.int32)                                                                             
for j in range(maximum_iterations):
    outputs, next_state, next_inputs, finished = decoder_instance.step(j,inputs,state)
    inputs = next_inputs
    state = next_state
    outputs = np.expand_dims(outputs.sample_id,axis = -1)
    predictions = np.append(predictions, outputs, axis = -1)