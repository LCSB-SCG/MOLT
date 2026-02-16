import logging
from src.preprocessing.DataPreProcessor import DataPreProcessor
from src.labels.InstanceLabeler import InstanceLabeler
from src.general.Cohort import Cohort
from src.general.Config import Config


class PlaqueTrackingPipeline:

    def __init__(self, config: Config):
        self.config = config
        self.cohort = Cohort(config=config)

    def update(self):
        self.cohort = self.cohort.update()

    def run_pipeline(self):
        logging.info("1. Running the DataPreprocessor")
        data_preprocessor = DataPreProcessor(config=self.config)
        data_preprocessor.preprocess_data()

        logging.info("2. Running the Instancelabeler")
        instance_labeler = InstanceLabeler(config=self.config, cohort=self.cohort)
        instance_labeler.run()
        self.update()

        # logging.info("3. Running the Registration")
        # registrator = GreedyRegistrator(config=self.config, cohort=self.cohort)
        # registrator.run()
        # self.update()

        # logging.info("4. Running the Instancematcher")
        # matcher = InstanceMatcher(config=self.config, cohort=self.cohort)
        # matcher.run()
        # self.update()
