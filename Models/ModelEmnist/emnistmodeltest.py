import sys, os
sys.path.append(os.getcwd())
from AllFunctions import *
from AllPaths import *

label_dictionary = CreateDictionary()
digit = predictValue(RecognizedCharactersTestPath, EMnistModelPath)
print(label_dictionary[digit])