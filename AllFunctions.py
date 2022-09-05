import os, shutil, cv2, tensorflow
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from numpy import argmax
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import np_utils
from AllPaths import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from sklearn.metrics import confusion_matrix

def load_image(filename):	# load and prepare the image
        img = load_img(filename, grayscale=True, target_size=(28, 28)) 	# load the image
        img = img_to_array(img) # convert to array
        img = img.reshape(1, 28, 28, 1)	# reshape into a single sample with 1 channel
        img = img.astype('float32') / 255.0	# prepare pixel data
        return img

def noise_removal(image):
    kernel = np.ones((7, 7), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel) # Morphological Gradient
    image = cv2.medianBlur(image, 3)
    return (image)

def count_files(rootdir): #counts the number of files in each subfolder in a directory
        numberoffiles = int(len([name for name in os.listdir(rootdir) if os.path.isfile(os.path.join(rootdir, name))]))
        return numberoffiles


def CreateFolders(): #Create all folders if they doesn't exist
    if not os.path.exists(folderPath):
            os.makedirs(folderPath)
    if not os.path.exists(createdFolderPath):
            os.makedirs(createdFolderPath)
    if not os.path.exists(RecognizedCharactersPath):
            os.makedirs(RecognizedCharactersPath)
    if os.path.exists(Temporarypath): #Remove the temporary folder if it exists to delete all the images from pervious run
        shutil.rmtree(Temporarypath, ignore_errors=True)
    if not os.path.exists(Temporarypath): #Used to store the images temporarly before moving them to the other folders
        os.makedirs(Temporarypath)


def CreateLettersAndDigitsFolders(RecognizedCharactersPath): #Create all folders and get their length by storing them in dictionary (used to name images when saving them next)
    character = 'A'
    NumberOfFiles = {} #empty dictionary
    CurrentDirectory = os.getcwd()
    for i in range(26): #26 characters
            CurrentFolder = RecognizedCharactersPath + '/' + character
            if not os.path.exists(CurrentFolder):
                    os.makedirs(CurrentFolder)
            NumberOfFiles[character] = count_files(CurrentDirectory + '/' + CurrentFolder)
            character = chr(ord(character) + 1) #Increment the character
            
    for i in range(10): #10 digits
            CurrentFolder = RecognizedCharactersPath + '/' + str(i)
            if not os.path.exists(CurrentFolder):
                    os.makedirs(CurrentFolder)
            NumberOfFiles[str(i)] = count_files(CurrentDirectory + '/' + CurrentFolder)
    return NumberOfFiles


def compare(rect1, rect2): # Sort the bounding boxes from left to right, top to bottom. sort by Y first, and then sort by X if Ys are similar
    if abs(rect1[1] - rect2[1]) > 10:
        return rect1[1] - rect2[1]
    else:
        return rect1[0] - rect2[0]


def loadmnistDataset():
	(X_train, y_train), (X_test, y_test) = mnist.load_data()	# load dataset
	X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))	# reshape dataset to have a single channel
	X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
	return (X_train, y_train), (X_test, y_test)

def load_dataset():	# load train and test dataset
	(X_train, y_train), (X_test, y_test) = loadmnistDataset()
	y_train = to_categorical(y_train)	# one hot encode target values
	y_test = to_categorical(y_test)
	return X_train, y_train, X_test, y_test

def prep_pixels(train, test):	# scale pixels
	train_norm = train.astype('float32') / 255.0	# convert from integers to floats and normalize to range 0-1
	test_norm = test.astype('float32') / 255.0
	return train_norm, test_norm	# return normalized images

def getModel(modelPath):
        return tensorflow.keras.models.load_model(modelPath)

def setUpperAndLowerBound(image): # Set lower bound and upper bound criteria for characters
    total_pixels = image.shape[0] * image.shape[1]
    lower = total_pixels // 1000
    upper = total_pixels // 5
    return lower, upper

def CreateSubplot(image, totalimagenumbers, imagenumberonplot, text):
    numberofcolumns = 2
    image = cv2.resize(image, (2500, 1700))
    plt.subplot(totalimagenumbers, numberofcolumns, imagenumberonplot)
    plt.title(text, fontsize = 22)
    plt.imshow(image)
    return imagenumberonplot + 1

def CreateConfusionMatrix(model, X_test, y_test, path):
    y_predicted = model.predict(X_test)
    predicted = [np.argmax(i) for i in y_predicted]
    cm = confusion_matrix(y_test, predicted)
    plt.figure(figsize = (18,10))
    sn.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.savefig(path)
    plt.show()

def CreateDictionary():
    label_map = pd.read_csv(BalancedMappingTextFile, delimiter = ' ', index_col=0, header=None, squeeze=True)
    label_dictionary = {}
    for index, label in enumerate(label_map):
        label_dictionary[index] = chr(label)
    return label_dictionary

def predictValue(imagePath, modelPath):# load an image and predict the class
    img = load_image(imagePath)	# load the image
    model = getModel(modelPath)	# load model
    predict_value = model.predict(img)	# predict the class
    digit = argmax(predict_value)
    return digit


def readEmnistCSV(path, number_of_classes, W, H):
    test_df = pd.read_csv(path, header=None)
    X_test = test_df.loc[:, 1:]
    y_test = test_df.loc[:, 0]
    X_test = np.apply_along_axis(reshape_and_rotate, 1, X_test.values)
    y_test = np_utils.to_categorical(y_test, number_of_classes)
    X_test = X_test.astype('float32') / 255
    X_test = X_test.reshape(-1, W, H, 1)
    return X_test, y_test

def reshape_and_rotate(image):
    W = 28
    H = 28
    image = image.reshape(W, H)
    image = np.fliplr(image)
    image = np.rot90(image)
    return image

# plot accuracy and loss
def plotgraph(epochs, acc, val_acc):
    # Plot training & validation accuracy values
    plt.plot(epochs, acc, 'b')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()