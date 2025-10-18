# entrenar_modelo.py — Fase 1: Preparación del Dataset Visual (CIFAR-10)

from skimage import color
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# 1. Cargar datos CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

# 2. Convertir imágenes a escala de grises y darles forma
x_train_gray = color.rgb2gray(x_train).reshape(-1, 32, 32, 1)
x_test_gray = color.rgb2gray(x_test).reshape(-1, 32, 32, 1)

# 3. One-hot encoding de etiquetas
num_classes = 10
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Exploración básica
print(f"x_train shape: {x_train_gray.shape}")
print(f"x_test shape: {x_test_gray.shape}")
print(f"y_train shape: {y_train_cat.shape}")
print(f"y_test shape: {y_test_cat.shape}")
