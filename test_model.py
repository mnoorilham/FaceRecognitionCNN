import os
import cv2
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input

from keras.utils import to_categorical

# Generate Dateset (random angka dengan 5 class data)
def generate_dataset(size, classes=5, noise=10.5):
    # Generate random datapoints
    labels = np.random.randint(0, classes, size)
    x1 = (np.random.rand(size) + labels) / classes
    x2 = x1 ** 2 + np.random.rand(size) * noise

    # Reshape data in order to merge them
    x1 = x1.reshape(size, 1)
    x2 = x2.reshape(size, 1)
    labels = labels.reshape(size, 1)

    # Merge the data
    data = np.hstack((x1, x2, labels))
    return data

dataset = generate_dataset(500)

print(dataset);

# plt.scatter(dataset[:, 0], dataset[:, 1], c=dataset[:,2], s=5, cmap='rainbow')
# plt.grid()
# plt.show()

X = dataset[:, :2]
Y = dataset[:, 2]
# print(X.shape, Y.shape);


# Encoding Label & Kategori
le = LabelEncoder()
le.fit(Y)

labels = le.classes_

print("Y :", Y[0])

Y = le.transform(Y)
print("Y (label encoding):", Y[0])

Y = to_categorical(Y)
print("Y (categorical):", Y[0])

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(X,   # input data
                                                    Y,   # target/output data
                                                    test_size=0.25,
                                                    random_state=42)

print(x_train.shape, y_train.shape, x_test.shape,  y_test.shape)

# Buat Model
def simple_model(input_dim):
    model = Sequential()

    model.add(Dense(64,
                    activation='relu',
                    input_shape=(input_dim,)))
    model.add(Dense(128, activation='relu', ))
    model.add(Dense(32, activation='relu', ))
    model.add(Dense(5))  # equal to number of classes
    model.add(Activation("sigmoid"))

    # print model network
    model.summary()

    # config model : add optimizer, loss & metrics
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Training Model
input_dim = x_train.shape[1]

EPOCHS = 100
BATCH_SIZE = 64

model = simple_model(input_dim)

history = model.fit(x_train,
                    y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_split=0.25   # 25% of train dataset will be used as validation set
                    )


# Buat evaluasinya
def evaluate_model_(history):
    names = [['accuracy', 'val_accuracy'],
             ['loss', 'val_loss']]
    for name in names:
        fig1, ax_acc = plt.subplots()
        plt.plot(history.history[name[0]])
        plt.plot(history.history[name[1]])
        plt.xlabel('Epoch')
        plt.ylabel(name[0])
        plt.title('Model - ' + name[0])
        plt.legend(['Training', 'Validation'], loc='lower right')
        plt.grid()
        plt.show()


evaluate_model_(history)