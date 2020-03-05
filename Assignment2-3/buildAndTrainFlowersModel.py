#Dataset Augmentation?
#Keras 
#Map, flat map?, mat. In preprossing: create new dataset, randomly shuffle, flip, reducing the brightness

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from preprocessDefinition import preprocess
from functools import partial

dataset, info = tfds.load("oxford_flowers102", as_supervised=True, with_info=True)
dataset_size = info.splits["train"].num_examples # 3670
class_names = info.features["label"].names # ["dandelion", "daisy", ...]
n_classes = info.features["label"].num_classes # 5

test_set = tfds.load('oxford_flowers102', split='test',as_supervised=True) # first 10% of train data
valid_set = tfds.load('oxford_flowers102', split='validation',as_supervised=True) # 10% to 25%
train_set = tfds.load('oxford_flowers102', split='train',as_supervised=True) #last 75% of train data

batch_size = 32
train_set = train_set.shuffle(1000).repeat()
train_set = train_set.map(partial(preprocess, randomize=True)).batch(batch_size).prefetch(1)
valid_set = valid_set.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set.map(preprocess).batch(batch_size).prefetch(1)

base_model = keras.applications.xception.Xception(weights="imagenet", include_top=False)
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation="softmax")(avg)
model = keras.Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False
optimizer = keras.optimizers.SGD(lr=0.2, momentum=0.9, decay=0.01)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
history = model.fit(train_set,
                    steps_per_epoch=int(0.75 * dataset_size / batch_size),
                    validation_data=valid_set,
                    validation_steps=int(0.15 * dataset_size / batch_size),
                    epochs=5,
                    )

for layer in base_model.layers:
    layer.trainable = True
optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True, decay=0.001)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
checkpoint_cb = keras.callbacks.ModelCheckpoint("flowersModel.h5", save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
history = model.fit(train_set,
                    steps_per_epoch=int(0.75 * dataset_size / batch_size),
                    validation_data=valid_set,
                    validation_steps=int(0.15 * dataset_size / batch_size),
                    epochs=40,
                    callbacks=[early_stopping_cb, checkpoint_cb]
                    )

