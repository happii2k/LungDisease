import sys

from src.cloud_storage.azure_syncer import AzureBlobSync
from src.constant import *
from src.entity.artifacts_config import DataIngestionArtifact
from src.entity.config import DataIngestionConfig
from src.exception import XRayException
from src.logger import logging


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

        self.blob = AzureBlobSync()

    def get_data_from_blob(self) -> None:
        try:
            logging.info("Entered the get_data_from_s3 method of Data ingestion class")

            self.blob.sync_folder_from_blob(
                folder=self.data_ingestion_config.blob_data_folder,
                container_name=self.data_ingestion_config.container_name,
                connection_string=self.data_ingestion_config.connection_string,
            )

            logging.info("Exited the get_data_from_s3 method of Data ingestion class")

        except Exception as e:
            raise XRayException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logging.info(
            "Entered the initiate_data_ingestion method of Data ingestion class"
        )

        try:
            self.get_data_from_blob()

            data_ingestion_artifact: DataIngestionArtifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.train_data_path,
                test_file_path=self.data_ingestion_config.test_data_path,
            )

            logging.info(
                "Exited the initiate_data_ingestion method of Data ingestion class"
            )

            return data_ingestion_artifact

        except Exception as e:
            raise XRayException(e, sys)

if __name__ == "__main__":
    data_ingestion = DataIngestion(DataIngestionConfig())
    data_ingestion.get_data_from_blob()
