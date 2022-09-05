#FolderPaths
folderPath = "Images"
createdFolderPath = 'CreatedFolder/'
Temporarypath = createdFolderPath + "TemporaryFolder"
ResultsPath = createdFolderPath + 'Results.png'
confusionMatrixpath = createdFolderPath + 'confusion_matrix_mnist.png'
confusionMatrixEmnistpath = createdFolderPath + 'confusion_matrix_emnist.png'
RecognizedCharactersPath =  createdFolderPath + 'RecognizedCharacters/'
RecognizedCharactersTestPath = RecognizedCharactersPath + '8/8_1.png'

AllModels = 'Models/'
emnistModel = AllModels + 'ModelEmnist/'
MnistModel = AllModels + 'ModelMnist/'
MnistModelPath = MnistModel + 'mnist_model.h5'
EMnistModelPath = emnistModel + 'emnist_model.h5'
BalancedMappingTextFile = emnistModel + 'emnist-balanced-mapping.txt'
emnistBalancedTrain = emnistModel + 'emnist-balanced-train.csv'
emnistBalancedTest = emnistModel + 'emnist-balanced-test.csv'

imagePath = folderPath + '/Image.png'