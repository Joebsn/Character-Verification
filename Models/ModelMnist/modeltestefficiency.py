import sys, os
sys.path.append(os.getcwd())
from AllFunctions import *
from AllPaths import *

def EvaluateModel():	# Evaluate the model
	(X_train, y_train), (X_test, y_test) = loadmnistDataset()	# load dataset
	X_train, X_test = prep_pixels(X_train, X_test)	# prepare pixel data
	model = getModel(MnistModelPath)
	CreateConfusionMatrix(model, X_test, y_test, confusionMatrixpath)
	y_train = to_categorical(y_train)	# one hot encode target values
	y_test = to_categorical(y_test)
	_, acc = model.evaluate(X_test, y_test, verbose=0)	# evaluate model on test dataset
	print('> %.3f' % (acc * 100.0))

EvaluateModel()