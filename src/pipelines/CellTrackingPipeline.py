import logging
from src.labels.InstanceLabeler import InstanceLabeler
from src.labels.InstanceMatcher import InstanceMatcher
from src.registration.GreedyRegistrator import GreedyRegistrator
from src.datasets.CellTrackingDataset import CellTrackingDataset
from src.general.Cohort import Cohort
from src.general.Config import Config


class CellTrackingPipeline:

    def __init__(self, config: Config):
        self.config = config
        self.cohort = Cohort(config=config)
        self.dataset = CellTrackingDataset(config=config)

    def update(self):
        self.dataset = self.dataset.update()
        self.cohort = self.cohort.update()

    def run_pipeline(self):
        logging.info("1. Running the Instancelabeler")
        instance_labeler = InstanceLabeler(config=self.config, cohort=self.cohort)
        instance_labeler.run()
        self.update()

        logging.info("2. Running the Registration")
        registrator = GreedyRegistrator(config=self.config, cohort=self.cohort)
        registrator.run()
        self.update()

        logging.info("3. Running the Instancematcher")
        matcher = InstanceMatcher(config=self.config, cohort=self.cohort)
        matcher.run()
        self.update()

        logging.info("4. Running the evaluation")
        self.dataset.run_evaluation()
