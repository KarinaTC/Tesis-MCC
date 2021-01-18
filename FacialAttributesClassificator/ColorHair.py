#  Tensorflow version 2.1
#  Classification Faces by color hair
#  Dataset CelabA pickle,


import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers, layers, models
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix


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
    model.add(layers.Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=(64, 64, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(128, (5, 5), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, activation='softmax'))
	# compile model
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    return model

# DATA EXPLORATION
dt = "D:/FaceDataHair.pkl"
dbfile = open(dt, 'rb')
data = pickle.load(dbfile)
dbfile.close()

dt = "D:/FaceDataHair_label.pkl"
dbfile = open(dt, 'rb')
label = pickle.load(dbfile)
dbfile.close()

datadf = pd.DataFrame(label, columns=['Bald','Brown_Hair','Black_Hair','Blond_Hair','Gray_Hair'])
datadf.head(5)

print(datadf['Bald'].value_counts())
print(datadf['Brown_Hair'].value_counts())
print(datadf['Black_Hair'].value_counts())
print(datadf['Blond_Hair'].value_counts())
print(datadf['Gray_Hair'].value_counts())

print('Shape images: ', data['Images'].shape)
X = data['Images']
print(X.min())
print(X.max())

# PREPROCESSING
# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.30)
print('X_train shape: ', X_train.shape)
print('X_test shape: ', X_test.shape)

unq_rows, count = np.unique(y_train,axis=0, return_counts=1)
out = {tuple(i):j for i,j in zip(unq_rows,count)}
counts_elements = [out[(1.0, 0.0, 0.0, 0.0, 0.0)], out[(0.0, 1.0, 0.0, 0.0, 0.0)],
                  out[(0.0, 0.0,1.0, 0.0, 0.0)], out[(0.0, 0.0, 0.0, 1.0, 0.0)],
                  out[(0.0, 0.0, 0.0, 0.0, 1.0)]]

unq_rows, count = np.unique(y_test,axis=0, return_counts=1)
out2 = {tuple(i):j for i,j in zip(unq_rows,count)}
counts_elements2 = [out2[(1.0, 0.0, 0.0, 0.0, 0.0)], out2[(0.0, 1.0, 0.0, 0.0, 0.0)],
                  out2[(0.0, 0.0,1.0, 0.0, 0.0)], out2[(0.0, 0.0, 0.0, 1.0, 0.0)],
                  out2[(0.0, 0.0, 0.0, 0.0, 1.0)]]

n_groups = 5

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
plt.xticks(index+0.17, ('Calvo', 'Castaño','Negro','Rubio','Gris'))
plt.legend()

plt.tight_layout()
plt.show()

plot_dataset(5,X_train,y_train)

# TRAIN MODEL
cnn = define_model()
cnn.summary()

checkpoint = keras.callbacks.ModelCheckpoint("D:/best_model_Hair.hdf5", monitor='loss', verbose=1,save_best_only=True, mode='auto', period=1)
history = cnn.fit(X_train, y_train, epochs=25,validation_data=(X_test, y_test), callbacks=[checkpoint])

# ACCURACY GRAPH
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

# LOSS GRAPH
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0.05, 1.5])
plt.legend(loc='lower right')

# EVALUATE MODEL
test_loss, test_acc = cnn.evaluate(X_test,  y_test, verbose=2)
print(test_acc)
y_pred = cnn.predict_classes(X_test)
y_pred

y_test_label = np.argmax(y_test, axis=1, out=None)
y_pred_onehot  = tf.one_hot(y_pred, 5)


print('Metricas de clasificación')
print(classification_report(y_test_label,y_pred))
print('Matriz de confusión')
print(confusion_matrix(y_test_label,y_pred))

plot_dataset(10,X_test,y_pred)