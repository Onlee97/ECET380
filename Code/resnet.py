from sklearn.datasets import load_sample_image
import tensorflow as tf
from tensorflow import keras
import numpy as np
# Load sample images
china = load_sample_image("china.jpg") / 255
flower = load_sample_image("flower.jpg") / 255
images = np.array([china, flower])
batch_size, height, width, channels = images.shape


model = keras.applications.resnet50.ResNet50(weights="imagenet")
images_resized = tf.image.resize(images, [224, 224])
#scale in next line if you have normalized (e.g. are using the images loaded earlier in book)
inputs = keras.applications.resnet50.preprocess_input(images_resized * 255)
# classify your inputs
Y_proba = model.predict(inputs)
# map the top classes back to labels
top_K = keras.applications.resnet50.decode_predictions(Y_proba, top=3)
for image_index in range(len(images)):
	print("Image #{}".format(image_index))
	for class_id, name, y_proba in top_K[image_index]:
		print(" {} - {:12s} {:.2f}%".format(class_id, name, y_proba * 100))
		print()