import numpy as np
import pandas as pd
import os

from ModelLoader import ModelLoader
from PhotoManager import PhotoManager
from PlotCreator import PlotCreator
from FolderManager import FolderManager

from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import layers
from keras.layers import *
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class EmnistModelTrain:

    def __init__(self):
        self.emnist_train_csv = ModelLoader.models_folder + 'csv/emnist-train.csv'
        self.emnist_test_csv = ModelLoader.models_folder + 'csv/emnist-test.csv'
        self.emnist_model_path = ModelLoader.models_folder + 'emnist_model.h5'
        self.label_dictionary = ModelLoader.create_dictionary()
        self.width = 28
        self.height = 28

    def evaluate_model(self):
        if not os.path.exists(self.emnist_model_path):
            self.__train_model()
 
        X_test, y_test = self.__prepare_values(self.emnist_test_csv)
        model = self.get_emnist_model()
        _, acc = model.evaluate(X_test, y_test)
        return acc * 100.0

    def __train_model(self):
        self.X_train, self.y_train = self.__prepare_values(self.emnist_train_csv)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size= 0.1, random_state=88) 
        self.model = self.__define_model()
        self.__fit_model()

    def __prepare_values(self, path):
        df = pd.read_csv(path, header=None)
        x = df.loc[:, 1:]
        y = df.loc[:, 0]
        x = x.astype('float32') / 255
        # np.apply_along_axis returns a numpy array, x is not a pandas.DataFrame anymore
        x = np.apply_along_axis(PhotoManager.reshape_and_rotate, 1, x.values)
        x = x.reshape(-1, self.width, self.height, 1)
        self.number_of_classes = y.nunique()
        y = to_categorical(y, self.number_of_classes)
        return x, y

    def __define_model(self):
        model = Sequential()
        model.add(layers.Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(self.width, self.height, 1)))
        model.add(layers.MaxPool2D(strides=2))
        model.add(layers.Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
        model.add(layers.MaxPool2D(strides=2))
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(84, activation='relu'))
        model.add(layers.Dense(self.number_of_classes, activation='softmax'))
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    def __fit_model(self):
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
        mcp_save = ModelCheckpoint(self.emnist_model_path, save_best_only=True, monitor='val_loss', verbose=1, mode='auto')
        history = self.model.fit(self.X_train, self.y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.1, callbacks=[early_stopping, mcp_save])
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1,len(acc)+1)
        PlotCreator.plot_graph(epochs, acc, val_acc, FolderManager.result_folder_path + "value_accuracy.png")
        PlotCreator.plot_graph(epochs, loss, val_loss, FolderManager.result_folder_path + "value_loss.png")

    def get_emnist_model(self):
        return ModelLoader.get_model(self.emnist_model_path)