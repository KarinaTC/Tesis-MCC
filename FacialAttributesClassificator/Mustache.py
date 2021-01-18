#  Tensorflow version 2.1
#  Classification Faces with Mustache
#  Dataset CelabA pickle,   dictionary with keys [Images,Mustache]

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import classification_report, confusion_matrix


def plot_dataset(n,X,y):
    '''
    Visualize the dataset with their labels
    param n: Number of images
    param X: Images
    param y: Labels
    '''
    plt.figure(figsize=(10,10))
    for i in range(n*n):
        plt.subplot(n,n,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X[i])
        plt.xlabel(y[i])
        plt.tight_layout()
    plt.show()


# Define CNN model
def define_model():
    '''
    Define Convolutional Neural Network
    return model: Convolutional Neural Network
    '''
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(64, 64, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
	# compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# DATA EXPLORATION
dt = "D:/FaceDataMustache.pkl"
dbfile = open(dt, 'rb')
data = pickle.load(dbfile)
dbfile.close()

print('Shape images: ', data['Images'].shape)

datadf = pd.DataFrame(data, columns=['Mustache'])
datadf['Young'].value_counts()

X = data['Images']
y = data['Mustache']
y = np.array(y)

# PREPROCESSING
# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
print('X_train shape: ', X_train.shape)
print('X_test shape: ', X_test.shape)

unique_elements, counts_elements = np.unique(y_train,return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements)))

unique_elements, counts_elements2 = np.unique(y_test,return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements2)))

n_groups = 2

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, counts_elements, bar_width,
alpha=opacity,
color='b',
label='Entrenamiento')

rects2 = plt.bar(index + bar_width, counts_elements2, bar_width,
alpha=opacity,
color='g',
label='Prueba')

plt.xlabel('Clases')
plt.ylabel('Frecuencia')
plt.xticks(index+0.17, ('Sin bigote', 'Bigote'))
plt.legend()

plt.tight_layout()
plt.show()

plot_dataset(5,X_train,y_train)

# TRAIN MODEL
cnn = define_model()
cnn.summary()
checkpoint = keras.callbacks.ModelCheckpoint("D:/best_model_Mustache.hdf5", monitor='loss', verbose=1,save_best_only=True, mode='auto')
history = cnn.fit(X_train, y_train, epochs=20,validation_data=(X_test, y_test), callbacks=[checkpoint])

# ACCURACY GRAPH
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.845, 1])
plt.legend(loc='lower right')

# LOSS GRAPH
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0.0, 0.2])
plt.legend(loc='lower right')

# EVALUATE MODEL
test_loss, test_acc = cnn.evaluate(X_test,  y_test, verbose=2)
print(test_acc)

y_pred = cnn.predict_classes(X_test)
plot_dataset(10,X_test,y_pred)

print('Matriz de confusión')
print(confusion_matrix(y_test,y_pred))
print("----------------------------------------------------- ")
print('Metricas de clasificación')
print(classification_report(y_test,y_pred))