#Dataset Augmentation?
#Keras 
#Map, flat map?, mat. In preprossing: create new dataset, randomly shuffle, flip, reducing the brightness

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from preprocessDefinition import preprocess

model=tf.keras.models.load_model('flowersModel.h5')
evalset,info = tfds.load(name='oxford_flowers102', split='test',as_supervised=True,with_info=True)
evalPipe=evalset.map(preprocess,num_parallel_calls=16).batch(128).prefetch(1)

for feats,lab in evalPipe.unbatch().batch(6000).take(1):
	probPreds=trainedModel1.predict(feats)

top1err=tf.reduce_mean(keras.metrics.sparse_top_k_categorical_accuracy(lab,probPreds,k=1))
top5err=tf.reduce_mean(keras.metrics.sparse_top_k_categorical_accuracy(lab,probPreds,k=5))
top10err=tf.reduce_mean(keras.metrics.sparse_top_k_categorical_accuracy(lab,probPreds,k=10))