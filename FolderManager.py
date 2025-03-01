import os, shutil

class FolderManager:

    result_folder_path = 'Results/'
    temporary_path = result_folder_path + "tmp/"

    @staticmethod
    def create_folders():
        if os.path.exists(FolderManager.result_folder_path):
            shutil.rmtree(FolderManager.result_folder_path, ignore_errors=True)
        if not os.path.exists(FolderManager.result_folder_path):
            os.makedirs(FolderManager.result_folder_path)
        if not os.path.exists(FolderManager.temporary_path):
            os.makedirs(FolderManager.temporary_path)

    @staticmethod
    def delete_tmp_folder():
        if os.path.exists(FolderManager.temporary_path):
            shutil.rmtree(FolderManager.temporary_path, ignore_errors=True)