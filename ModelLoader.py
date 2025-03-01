import pandas as pd
import tensorflow

from PhotoManager import PhotoManager

from numpy import argmax

class ModelLoader:

    models_folder = 'Models/'

    @staticmethod
    def get_model(model_path):
        return tensorflow.keras.models.load_model(model_path)
    
    @staticmethod
    def scale_pixels(train, test):
        train_norm = train.astype('float32') / 255.0
        test_norm = test.astype('float32') / 255.0
        return train_norm, test_norm
    
    @staticmethod
    def predict_value(image_path, model_path):
        img = PhotoManager.load_image(image_path)
        model = ModelLoader.get_model(model_path)
        predict_value = model.predict(img)
        digit = argmax(predict_value)
        return digit
    
    @staticmethod
    def create_dictionary():
        balanced_mapping_text_file = ModelLoader.models_folder + 'emnist-balanced-mapping.txt'
        label_map = pd.read_csv(balanced_mapping_text_file, delimiter = ' ', index_col=0, header=None)
        label_map = label_map.squeeze()
        label_dictionary = {}
        for index, label in enumerate(label_map):
            label_dictionary[index] = chr(label)
        return label_dictionary