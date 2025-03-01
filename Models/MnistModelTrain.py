import os

from FolderManager import FolderManager
from PlotCreator import PlotCreator
from ModelLoader import ModelLoader

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

class MnistModelTrain:

	def __init__(self):
		self.confusion_matrix_path = FolderManager.result_folder_path + "confusion_matrix_mnist.png"
		self.mnist_model_path = ModelLoader.models_folder + 'mnist_model.h5'
		self.X_train = None
		self.y_train = None
		self.X_test = None
		self.y_test = None
		self.encoded_y_train = None
		self.encoded_y_test = None
		self.scaled_X_train = None
		self.scaled_X_test = None

	def evaluate_model(self):
		if not os.path.exists(self.mnist_model_path):
			self.__train_model()
		
		elif self.X_train is None:
			self.__prepare_values()

		model = self.get_mnist_model()
		_, acc = model.evaluate(self.scaled_X_test, self.encoded_y_test, verbose=0)
		return acc * 100.0

	def __train_model(self):
		self.model = self.__define_model()
		self.__prepare_values()
		self.model.fit(self.scaled_X_train, self.encoded_y_train, epochs=10, batch_size=32, verbose=0)
		self.model.save(self.mnist_model_path)
		PlotCreator.create_confusion_matrix(self.model, self.scaled_X_test, self.y_test, self.confusion_matrix_path)
	
	def __define_model(self):
		model = Sequential()
		model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
		model.add(MaxPooling2D((2, 2)))
		model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
		model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
		model.add(MaxPooling2D((2, 2)))
		model.add(Flatten())
		model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
		model.add(Dense(10, activation='softmax'))
		opt = SGD(learning_rate=0.01, momentum=0.9)
		model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
		return model

	def __prepare_values(self):
		self.__load_mnist_dataset_and_reshape_to_have_single_channel()
		self.__one_hot_encode_target_values()
		self.scaled_X_train, self.scaled_X_test = ModelLoader.scale_pixels(self.X_train, self.X_test)

	def __load_mnist_dataset_and_reshape_to_have_single_channel(self):
		(self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
		self.X_train = self.X_train.reshape((self.X_train.shape[0], 28, 28, 1))
		self.X_test = self.X_test.reshape((self.X_test.shape[0], 28, 28, 1))
	
	def __one_hot_encode_target_values(self):
		self.encoded_y_train = to_categorical(self.y_train)
		self.encoded_y_test = to_categorical(self.y_test)

	def get_mnist_model(self):
		return ModelLoader.get_model(self.mnist_model_path)