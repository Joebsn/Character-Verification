import sys, os
sys.path.append(os.getcwd())
from AllFunctions import *
from AllPaths import *
 
print(predictValue(RecognizedCharactersTestPath, MnistModelPath))