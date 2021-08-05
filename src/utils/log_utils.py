import logging
import pathlib
import sys


class LastPartFilter(logging.Filter):
    def filter(self, record):
        record.name_last = record.name.rsplit('.', 1)[-1]
        return True


def reset_logger(base_level=logging.DEBUG):
    logger = logging.getLogger('src')
    logger.setLevel(base_level)
    logger.handlers = []
    logger.propagate = False


def add_console_logger(level=logging.INFO):
    logger = logging.getLogger('src')
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    formatter = logging.Formatter(
        '%(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def add_file_logger(logfile, level=logging.DEBUG, clear_log=False):
    log_path = pathlib.Path(logfile)
    if log_path.is_file():
        if clear_log:
            open(log_path, "w").close()
    else:
        pathlib.Path(log_path.parent).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('src')
    fh = logging.FileHandler(log_path)
    fh.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    fh.addFilter(LastPartFilter())
    logger.addHandler(fh)
