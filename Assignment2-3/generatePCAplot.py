import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from preprocessDefinition import preprocess
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

test_set, info = tfds.load('oxford_flowers102', split='test',as_supervised=True, with_info=True)
dataset_size = info.splits["train"].num_examples
class_names = info.features["label"].names
n_classes = info.features["label"].num_classes

base_model = keras.applications.xception.Xception(weights="imagenet", include_top=False)
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
model = keras.Model(inputs=base_model.input, outputs=avg)

evalPipe = test_set.map(preprocess,num_parallel_calls=16).batch(128).prefetch(1)
for feature,label in evalPipe.unbatch().batch(1000).take(1):
	result=model.predict(feature)

print(result)

pca = PCA()
pca.fit(result)
var = pca.explained_variance_ratio_
print(var)
plt.plot((np.cumsum(var)))
plt.ylabel("Explained Variance")
plt.xlabel("Dimensions")
plt.savefig("explainedVariancePlot.png")
plt.show()


print("Done")