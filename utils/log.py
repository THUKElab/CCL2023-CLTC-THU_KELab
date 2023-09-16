import logging
import sys

LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s -   %(message)s"
LOG_DATE = "%m/%d/%Y %H:%M:%S"
LOG_FORMATTER = logging.Formatter(LOG_FORMAT, LOG_DATE)

logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATE,
    handlers=[logging.StreamHandler(sys.stdout)],
)

# from logging.handlers import RotatingFileHandler
# LOG_FILENAME = os.path.join(BASE_DIR, 'gec_sys.log')
# LOG_HANDLER = RotatingFileHandler(
#     filename=LOG_FILENAME,
#     mode="a",
#     maxBytes=50 * 1024 * 1024,
#     backupCount=5,
#     encoding="utf-8",
# )

# LOG_HANDLER = logging.StreamHandler()
# LOG_HANDLER.setFormatter(LOG_FORMATTER)


def get_logger(name, level=LOG_LEVEL):
    """
    Initialises and returns named Django logger instance.
    """
    named_logger = logging.getLogger(name=name,)
    named_logger.setLevel(level)

    # named_logger.addHandler(LOG_HANDLER)
    return named_logger
