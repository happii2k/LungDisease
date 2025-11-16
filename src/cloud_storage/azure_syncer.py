import os
import sys
from src.exception import XRayException
class AzureBlobSync:

    def sync_folder_to_blob(self, folder, container_name, connection_string):
        """
        Upload all files in a local folder to an Azure Blob Storage container.

        Parameters:
        - folder: local folder path to upload
        - container_name: Azure Blob container name
        - connection_string: Azure Storage account connection string

        Raises exceptions on failure.

        """
        try:
            command = (
                f'az storage blob upload-batch '
                f'--destination {container_name} '
                f'--source {folder} '
                f'--connection-string "{connection_string}"'
            )
            os.system(command)
        except Exception as e:
            raise XRayException(e , sys)


    def sync_folder_from_blob(self, folder, container_name, connection_string):
        """
        Download all blobs from an Azure Blob Storage container to a local folder.

        Parameters:
        - folder: local folder path to download blobs into
        - container_name: Azure Blob container name
        - connection_string: Azure Storage account connection string

        Raises exceptions on failure.
        """
        try:
            if not os.path.exists(folder):
                os.makedirs(folder)
            command = (
                f'az storage blob download-batch '
                f'--destination {folder} '
                f'--source {container_name} '
                f'--connection-string "{connection_string}"'
            )
            os.system(command)
        except Exception as e:
            raise XRayException(e , sys)

