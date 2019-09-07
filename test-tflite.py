import tensorflow as tf
from tensorflow import keras
import numpy as np

print(tf.__version__)

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

interpreter = tf.lite.Interpreter(model_path="mnist_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
print('input_details:', input_details)
output_details = interpreter.get_output_details()
print('output_details:', output_details)

input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]["index"], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]["index"])
output = np.squeeze(output_data)
print(output)

input_image = np.reshape(test_images[0], input_details[0]['shape'])
np.float32(input_image)
interpreter.set_tensor(input_details[0]['index'], input_image)

interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
output = np.squeeze(output_data)
print(output)
