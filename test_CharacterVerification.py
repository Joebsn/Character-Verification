import unittest

from ModelLoader import ModelLoader
from FolderManager import FolderManager
from CharacterVerification import CharacterVerification
from Models.MnistModelTrain import MnistModelTrain
from Models.EmnistModelTrain import EmnistModelTrain

class TestCharacterVerification(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        FolderManager.create_folders()

    @classmethod
    def tearDownClass(cls):
        FolderManager.delete_tmp_folder()

    images_folder = ModelLoader.models_folder + "Images/"

    def test_mnist(self):
        mnist_image_1 = "mnist_image1.png"
        mnist_image_2 = "mnist_image2.png"

        mnistModelTrain = MnistModelTrain()
        accuracy = mnistModelTrain.evaluate_model()
        self.assertTrue(accuracy > 95)

        characterVerification = CharacterVerification()
        model = mnistModelTrain.get_mnist_model()
        text = characterVerification.verify_image(model, self.images_folder + mnist_image_1, FolderManager.result_folder_path + mnist_image_1)
        self.__assert_on_text(text, "817")
        text = characterVerification.verify_image(model, self.images_folder + mnist_image_2, FolderManager.result_folder_path + mnist_image_2)
        self.__assert_on_text(text, "052")

    def test_emnist(self):
        emnist_image_1 =  "emnist_image1.png"
        emnist_image_2 =  "emnist_image2.png"

        emnistModelTrain = EmnistModelTrain()
        accuracy = emnistModelTrain.evaluate_model()
        self.assertTrue(accuracy > 85)

        characterVerification = CharacterVerification()
        model = emnistModelTrain.get_emnist_model()
        text = characterVerification.verify_image(model, self.images_folder + emnist_image_1, FolderManager.result_folder_path + emnist_image_1)
        self.__assert_on_text(text, "8ABW")
        text = characterVerification.verify_image(model, self.images_folder + emnist_image_2, FolderManager.result_folder_path + emnist_image_2)
        self.__assert_on_text(text, "LMA")

    def __assert_on_text(self, text, expectedText):
        self.assertEqual(text, expectedText, "Expected " + expectedText + " but got " + text)

if __name__ == '__main__':
    unittest.main()