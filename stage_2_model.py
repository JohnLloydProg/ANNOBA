import os
import pickle
import settings
import tensorflow as tf
import keras
import numpy as np
from keras import layers, models
from keras.callbacks import TensorBoard
import settings

features = []
labels = []

for file in os.listdir(os.path.join(settings.stage2_fileoutput, 'features')):
    with open(os.path.join(settings.stage2_fileoutput, f'features/{file}'), 'rb') as f:
        inp = f.read()
        features.extend(pickle.loads(inp))
for file in os.listdir(os.path.join(settings.stage2_fileoutput, 'labels')):
    with open(os.path.join(settings.stage2_fileoutput, f'labels/{file}'), 'rb') as f:
        inp = f.read()
        labels.extend(pickle.loads(inp))

testing_features = np.array(features[0:500], dtype=np.float32)
testing_labels = np.array(labels[0:500])
training_features = np.array(features[500:], dtype=np.float32)
training_labels = np.array(labels[500:])

testing_features = testing_features / 255.0
training_features = training_features / 255.0

for no_dense_layer in settings.stage2_no_dense_layer:
    for dense_layer_size in settings.stage2_dense_layer_size:
        for no_conv_layer in settings.stage2_no_conv_layer:
            for conv_layer_size in settings.stage2_conv_layer_size:
                name = f'Stage-Two_Dense-{no_dense_layer}-Nodes-{dense_layer_size}_Conv-{no_conv_layer}-Nodes-{conv_layer_size}'
                tensorboard = TensorBoard(log_dir=f'logs/stage2/{name}')

                model = models.Sequential()

                model.add(layers.Conv2D(conv_layer_size, (3, 3), activation='relu', input_shape=(settings.picture_height, settings.picture_width, 3)))
                model.add(layers.BatchNormalization())
                model.add(layers.MaxPool2D((2, 2)))

                for i in range(no_conv_layer-1):
                    model.add(layers.Conv2D(conv_layer_size, (3, 3), activation='relu'))
                    model.add(layers.BatchNormalization())
                    model.add(layers.MaxPool2D((2, 2)))

                model.add(layers.Dropout(0.2))

                model.add(layers.Flatten())

                for i in range(no_dense_layer):
                    model.add(layers.Dense(dense_layer_size, activation='relu'))

                model.add(layers.Dense(3, activation='softmax'))

                model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
                history = model.fit(training_features, training_labels, batch_size=5, epochs=5, validation_data=(testing_features, testing_labels))
                print(history)
