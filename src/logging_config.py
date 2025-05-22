import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logging(log_file='logs/app.log', level=logging.INFO):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    handler.setLevel(level)

    formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')
    handler.setFormatter(formatter)

    logging.getLogger().setLevel(level)
    logging.getLogger().addHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

def log_service_config():
    logger = logging.getLogger(__name__)
    logger.info("Starting captioning service.")
    logger.info("Environment Configuration:\n")
    logger.info(f"MODEL_NAME: {os.getenv('MODEL_NAME')}")
    logger.info(f"DEVICE: {os.getenv('DEVICE')}")
    logger.info(f"PORT: {os.getenv('PORT')}")
    logger.info(f"HOST: {os.getenv('HOST')}")
    logger.info(f"MAX_IMAGE_SIZE: {os.getenv('MAX_IMAGE_SIZE')}\n")
