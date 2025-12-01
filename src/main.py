import logging

from src.datasets.CellTrackingDataset import CellTrackingDataset
from src.general.Config import Config
from src.pipelines.CellTrackingPipeline import CellTrackingPipeline
from src.utils.common import logging_to_stdout


def main(config_path: str, download_cell_datsets: bool = False):
    # set the logging parameters:
    logging.basicConfig(level=logging.INFO)
    logging_to_stdout()

    if download_cell_datsets:
        logging.info("Setting up the datsets...")
        CellTrackingDataset.setup_datasets()

    # Laod the config and select the pipeline to run
    config = Config(config_path=config_path)

    if config["general"]["pipeline"] == "CellTracking":
        CellTrackingPipeline(config).run_pipeline()
    else:
        logging.warning("Pipeline not implemented, please select a correct Pipeline.")


if __name__ == "__main__":
    # config_path = input("Enter the path to the config file:")
    config_path = "/workspaces/MOLT/data/CellTracking/Fluo-N2DH-GOWT1/config.json"
    main(config_path=config_path, download_cell_datsets=False)
