import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

from sklearn.metrics import confusion_matrix

class PlotCreator:

    @staticmethod
    def create_subplot(image, nonoise, recognized_text, recognized_image_save_path):
        PlotCreator.__add_image(image, "Initial Image", 1)
        PlotCreator.__add_image(nonoise, "No Noise Image, recognized text: " + recognized_text, 2)
        plt.savefig(recognized_image_save_path)

    @staticmethod
    def __add_image(image, text, index):
        image = cv2.resize(image, (2500, 1700))
        plt.subplot(1, 2, index)
        plt.title(text, fontsize = 22)
        plt.imshow(image)

    @staticmethod
    def create_confusion_matrix(model, X_test, y_test, path):
        y_predicted = model.predict(X_test)
        predicted = [np.argmax(i) for i in y_predicted]
        cm = confusion_matrix(y_test, predicted)
        plt.figure(figsize = (18,10))
        sn.heatmap(cm, annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.savefig(path)

    @staticmethod
    def plot_graph(epochs, acc, val_acc, path):
        plt.plot(epochs, acc, 'b')
        plt.plot(epochs, val_acc, 'r')
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig(path)