import functools, cv2

from FolderManager import FolderManager
from ModelLoader import ModelLoader
from PhotoManager import PhotoManager
from PlotCreator import PlotCreator

from matplotlib.pyplot import figure

class CharacterVerification:

	def verify_image(self, model, image_path, recognized_image_save_path):
		figure(figsize=(12, 5), dpi=80)
		label_dictionary = ModelLoader.create_dictionary()
		image, thresh, nonoise, nonoiseinv, labels, mask, lower, upper = PhotoManager.remove_image_noise(image_path)
		mask = PhotoManager.loop_over_unique_components(mask, labels, thresh, lower, upper)
		cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		boundingBoxes = [cv2.boundingRect(c) for c in cnts]
		boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(PhotoManager.compare) )
		nonoise, image = PhotoManager.draw_rectangle_on_characters(image, nonoise, nonoiseinv, boundingBoxes)
		recognized_text = PhotoManager.get_recognized_text_in_folder(model, label_dictionary)
		PlotCreator.create_subplot(image, nonoise, recognized_text, recognized_image_save_path)
		return recognized_text