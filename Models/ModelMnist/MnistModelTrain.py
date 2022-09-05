from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import SGD
import sys, os
sys.path.append(os.getcwd())
from AllFunctions import *
from AllPaths import *

def define_model():	# define cnn model
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(learning_rate=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def TrainModel():
	trainX, trainY, testX, testY = load_dataset()	# load dataset
	trainX, testX = prep_pixels(trainX, testX)	# prepare pixel data
	model = define_model()	# define model
	model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=0)	# fit model
	model.save(MnistModelPath)	# save model
 
TrainModel()