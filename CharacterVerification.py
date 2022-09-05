import numpy as np
import matplotlib.pyplot as plt
import functools, glob, os, cv2
from matplotlib.pyplot import figure
from AllFunctions import *
from AllPaths import *

#model = getModel(MnistModelPath)
model = getModel(EMnistModelPath)

CreateFolders()
label_dictionary = CreateDictionary() #For character and letter recognition
NumberOfFiles = CreateLettersAndDigitsFolders(RecognizedCharactersPath)

#Count number of images
totalimagenumbers = len(os.listdir(folderPath))

imagenumberonplot = 1 #Used for the plt graph
figure(figsize=(15, 18), dpi=80)

for images in glob.iglob(f'{folderPath}/*'):
        #Text to recognize
        text_to_recognize = input("Enter the text you want to recognize: ") 
        text_to_recognize_length = len(text_to_recognize)

        image = cv2.imread(images) # Read the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Convert to grayscale
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU )[1] 
        nonoise = noise_removal(thresh)
        nonoiseinv = cv2.bitwise_not(nonoise)

        _, labels = cv2.connectedComponents(nonoiseinv)
        mask = np.zeros(thresh.shape, dtype="uint8") #initialize the mask to hold only the components we are interested in

        lower, upper = setUpperAndLowerBound(image)

        for (i, label) in enumerate(np.unique(labels)):  # Loop over the unique components
                if label != 0: # If this is not the background label
                    #construct the label mask to display only connected component for the current label
                    labelMask = np.zeros(thresh.shape, dtype="uint8")
                    labelMask[labels == label] = 255
                    numPixels = cv2.countNonZero(labelMask)
                    # If the number of pixels in the component is between lower bound and upper bound, add it to our mask
                    if numPixels > lower and numPixels < upper: 
                        mask = cv2.add(mask, labelMask)

        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find contours
        boundingBoxes = [cv2.boundingRect(c) for c in cnts] # Get bounding box for each contour

        boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(compare) )
        temp = 0
        for rect in boundingBoxes:
                x,y,w,h = rect  # Get the coordinates from the bounding box
                
                cv2.rectangle(nonoise, (x,y), (x+w,y+h), (150, 150, 150), 1)  # Show bounding box and prediction on image
                cv2.rectangle(image,(x,y), (x+w,y+h), (0, 250, 0), 5)   #Draw rectangle on the image itself around current character

                ROI = nonoiseinv[y:y+h, x:x+w] #Getting the image containing 1 character
                ROI = cv2.copyMakeBorder(ROI, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                TemporaryImagePath = Temporarypath + '/' + 'img{}.png'.format(str(temp))
                cv2.imwrite(TemporaryImagePath, ROI) #Saving the image in the temporary folder
                temp = temp + 1

        #Loop through all the pictures in the temporary folder and get the text recognized
        SameText = True
        index = 0
        recognizedfinaltext = ""
        for filename in os.scandir(Temporarypath):
                if filename.is_file():
                        loadedimage = load_image(filename.path) #load the image from the temporary file
                        image1 = cv2.imread(filename.path)
                        os.remove(filename.path) #Remove the file from the temporary folder

                        predict_values = model.predict(loadedimage) #Predict the value in the image
                        predict_value = np.argmax(predict_values)
                        predict_value = str(label_dictionary[predict_value])

                        recognizedfinaltext = recognizedfinaltext + predict_value

                        if predict_value.isdigit() == False: #check if it is a letter or a number
                                predict_value = predict_value.upper() #Folders names are uppercase, Getting the uppercase character so we save the image
        
                        NumberOfFiles[predict_value] = NumberOfFiles[predict_value] + 1
                        currentimagenumber =  NumberOfFiles[predict_value]
                        Imagepath = RecognizedCharactersPath + '/' + predict_value + '/' + predict_value + '_{}.png'.format(currentimagenumber)

                        cv2.imwrite(Imagepath, image1) #Saving the image

                        #Removing the spaces from the user input text
                        while index < text_to_recognize_length and text_to_recognize[index] == " ":
                                index = index + 1
                        if index >= text_to_recognize_length or text_to_recognize[index] != predict_value: #Check if it corresponds to the user input
                                SameText = False
                        index = index + 1

        print("The recognized text is: " + recognizedfinaltext)
        print("The text to recognize is : " + text_to_recognize)
        m = ""
        if SameText == True:
                print("Correct! The text to recognized entered is the same as the text in the image")
                m = "Same Text"
        else:
                print("Incorrect! The text to recognized entered is not the same as the text in the image")
                m = "Not Same Text"

        imagenumberonplot = CreateSubplot(image, totalimagenumbers, imagenumberonplot, "Initial Image")
        imagenumberonplot = CreateSubplot(nonoise, totalimagenumbers, imagenumberonplot, "No Noise Image, recognized text: " + recognizedfinaltext + ",  " + m)

plt.savefig(ResultsPath)
plt.show()
plt.waitforbuttonpress()
cv2.waitKey(0)
cv2.destroyAllWindows()