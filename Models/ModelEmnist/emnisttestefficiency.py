import pandas as pd
import sys, os
sys.path.append(os.getcwd())
from AllFunctions import *
from AllPaths import *

train_df = pd.read_csv(emnistBalancedTrain, header=None)
X_train = train_df.loc[:, 1:]
y_train = train_df.loc[:, 0]

number_of_classes = y_train.nunique()
test_df = pd.read_csv(emnistBalancedTest, header=None)
X_test, y_test = readEmnistCSV(emnistBalancedTest, number_of_classes, 28, 28)

model = getModel(EMnistModelPath)
_, acc = model.evaluate(X_test, y_test)	# evaluate model on test dataset
print('> %.3f' % (acc * 100.0))

CreateConfusionMatrix(model, X_test, test_df.loc[:, 0], confusionMatrixEmnistpath)