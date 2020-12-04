import logging
import os


def setup_logger(filepath=None):
    # Log object
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    # Set formatter
    formatter_file = logging.Formatter('[%(asctime)s %(levelname)s %(lineno)4s] %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    formatter_stream = logging.Formatter('[%(asctime)s %(lineno)4s] %(message)s', datefmt='%Y/%m/%d %H:%M')

    # FileHandler
    if not filepath:
        pass
    else:
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        file_handler = logging.FileHandler(filename=filepath)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter_file)
        logger.addHandler(file_handler)

    # StreamHandler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(formatter_stream)
    logger.addHandler(stream_handler)

    return logger
