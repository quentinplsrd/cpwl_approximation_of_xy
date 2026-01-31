import logging
import colorlog
import os

def setup_logging():
    """Set up logging with colorlog"""
    handler = colorlog.StreamHandler()
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(yellow)s%(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    handler.setFormatter(formatter)
    # Create a logger instance
    logger = colorlog.getLogger('tempregpy')
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    # Add a FileHandler to save logs to a file
    log_file = 'app.log'
    if os.path.exists(log_file):
        os.remove(log_file)
        logger.info(f"Existing '{log_file}' found and deleted.")

    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    return logger

# Initialize the logger
logger = setup_logging()