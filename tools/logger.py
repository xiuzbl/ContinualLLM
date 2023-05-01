import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


class Logger():
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def info(self, message, *args, **kwargs):
        self.logger.info(message)

    def error(self, message, *args, **kwargs):
        self.logger.error(message, *args, **kwargs)
