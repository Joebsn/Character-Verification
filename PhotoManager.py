import os
import cv2
import numpy as np

from FolderManager import FolderManager

from keras.preprocessing.image import load_img, img_to_array

class PhotoManager:

    @staticmethod
    def load_image(file_name):
        img = load_img(file_name, color_mode='grayscale', target_size=(28, 28))
        img = img_to_array(img)
        img = img.reshape(1, 28, 28, 1)	# reshape into a single sample with 1 channel
        img = img.astype('float32') / 255.0
        return img
    
    @staticmethod
    def reshape_and_rotate(image):
        W = 28
        H = 28
        image = image.reshape(W, H)
        image = np.fliplr(image)
        image = np.rot90(image)
        return image
    
    @staticmethod
    def compare(rect1, rect2): # Sort the bounding boxes from left to right, top to bottom.
        if abs(rect1[1] - rect2[1]) > 10:
            return rect1[1] - rect2[1]
        else:
            return rect1[0] - rect2[0]
    
    @staticmethod
    def remove_image_noise(image_path):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU )[1] 
        nonoise = PhotoManager.__remove_noise(thresh)
        nonoiseinv = cv2.bitwise_not(nonoise)
        _, labels = cv2.connectedComponents(nonoiseinv)
        mask = np.zeros(thresh.shape, dtype="uint8") #mask to hold only the components we are interested in
        lower, upper = PhotoManager.__set_upper_and_lower_bound(image)
        return image, thresh, nonoise, nonoiseinv, labels, mask, lower, upper
    
    @staticmethod
    def __remove_noise(image):
        kernel = np.ones((7, 7), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.medianBlur(image, 3)
        return (image)

    @staticmethod
    def __set_upper_and_lower_bound(image):
        total_pixels = image.shape[0] * image.shape[1]
        lower = total_pixels // 1000
        upper = total_pixels // 5
        return lower, upper

    @staticmethod
    def loop_over_unique_components(mask, labels, thresh, lower, upper):
        for (i, label) in enumerate(np.unique(labels)):
            if label != 0: # If this is not the background label, construct the label mask to display only connected component for the current label
                labelMask = np.zeros(thresh.shape, dtype="uint8")
                labelMask[labels == label] = 255
                numPixels = cv2.countNonZero(labelMask)
                if numPixels > lower and numPixels < upper: # If lower bound < number of pixels in the component < upper bound, add it to the mask
                    mask = cv2.add(mask, labelMask)
        return mask
    
    @staticmethod
    def draw_rectangle_on_characters(image, nonoise, nonoiseinv, boundingBoxes):
        temp = 0
        for rect in boundingBoxes:
            x, y, w, h = rect
            cv2.rectangle(nonoise, (x,y), (x+w,y+h), (150, 150, 150), 1) 
            cv2.rectangle(image, (x,y), (x+w,y+h), (0, 250, 0), 5)
            ROI = nonoiseinv[y:y+h, x:x+w]
            ROI = cv2.copyMakeBorder(ROI, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            TemporaryImagePath = FolderManager.temporary_path + '/' + 'img{}.png'.format(str(temp))
            cv2.imwrite(TemporaryImagePath, ROI)
            temp = temp + 1
        return nonoise, image
    
    @staticmethod
    def get_recognized_text_in_folder(model, label_dictionary):
        recognizedfinaltext = ""
        for filename in os.scandir(FolderManager.temporary_path):
            if filename.is_file():
                loadedimage = PhotoManager.load_image(filename.path)
                os.remove(filename.path)
                predict_values = model.predict(loadedimage)
                predict_value = np.argmax(predict_values)
                predict_value = str(label_dictionary[predict_value])
                recognizedfinaltext = recognizedfinaltext + predict_value
        
        return recognizedfinaltext