import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import layers
from keras.layers import *
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import sys, os
sys.path.append(os.getcwd())
from AllFunctions import *
from AllPaths import *


train_df = pd.read_csv(emnistBalancedTrain, header=None)
X_train = train_df.loc[:, 1:]
y_train = train_df.loc[:, 0]

label_dictionary = CreateDictionary() #For character and letter recognition

# Sample entry number 42
sample_image = X_train.iloc[42]
sample_label = y_train.iloc[42]

W = 28
H = 28

print("Label entry 42:", label_dictionary[sample_label])
plt.imshow(sample_image.values.reshape(W, H), cmap=plt.cm.gray)
plt.show()

print("Label entry 42:", label_dictionary[sample_label])
plt.imshow(reshape_and_rotate(sample_image.values), cmap=plt.cm.gray)
plt.show()

# note: np.apply_along_axis returns a numpy array, X_train is not a pandas.DataFrame anymore
X_train = np.apply_along_axis(reshape_and_rotate, 1, X_train.values)

sample_image = X_train[42]
sample_label = y_train.iloc[42]
print("Label entry 42:", label_dictionary[sample_label])
plt.imshow(sample_image.reshape(W, H), cmap=plt.cm.gray)
plt.show()

for i in range(100, 106):
    plt.subplot(390 + (i+1))
    plt.imshow(X_train[i], cmap=plt.cm.gray)
    plt.title(label_dictionary[y_train[i]])

X_train = X_train.astype('float32') / 255 #Scaling
number_of_classes = y_train.nunique()

y_train = np_utils.to_categorical(y_train, number_of_classes) #One hot encoding

# Reshape to fit model input shape
# Tensorflow (batch, width, height, channels)
X_train = X_train.reshape(-1, W, H, 1)

# Split 10% validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size= 0.1, random_state=88)

model = Sequential()

model.add(layers.Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(W, H, 1)))
model.add(layers.MaxPool2D(strides=2))
model.add(layers.Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
model.add(layers.MaxPool2D(strides=2))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(84, activation='relu'))
model.add(layers.Dense(number_of_classes, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
mcp_save = ModelCheckpoint(EMnistModelPath, save_best_only=True, monitor='val_loss', verbose=1, mode='auto')

history = model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1, validation_split=0.1, callbacks=[early_stopping, mcp_save])
 
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)

plotgraph(epochs, acc, val_acc) # Accuracy curve
plotgraph(epochs, loss, val_loss) # loss curve

model = load_model(EMnistModelPath) # Load emnsit model
model.summary()

y_pred = model.predict(X_val)

for i in range(10, 16):
    plt.subplot(380 + (i%10+1))
    plt.imshow(X_val[i].reshape(28, 28), cmap=plt.cm.gray)
    plt.title(label_dictionary[y_pred[i].argmax()])

for i in range(42, 48):
    plt.subplot(380 + (i%10+1))
    plt.imshow(X_val[i].reshape(28, 28), cmap=plt.cm.gray)
    plt.title(label_dictionary[y_pred[i].argmax()])

print(model.evaluate(X_val, y_val))

X_test, y_test = readEmnistCSV(emnistBalancedTest, number_of_classes, W, H)

print(model.evaluate(X_test, y_test))