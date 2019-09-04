import tensorflow as tf
from tensorflow import keras
import numpy as np

print(tf.__version__)

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax),
])
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)
model.evaluate(test_images, test_labels)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE] # tf.lite.Optimize.{DEFAULT,OPTIMIZE_FOR_SIZE,OPTIMIZE_FOR_LATENCY}
#converter.optimizations = []
tflite_model = converter.convert()
open("mnist_model.tflite", "wb").write(tflite_model)
