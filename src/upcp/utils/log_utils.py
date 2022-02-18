# Urban_PointCloud_Processing by Amsterdam Intelligence, GPL-3.0 license

import logging
import pathlib
import sys

BASE_NAME = 'upcp'
BASE_LEVEL = logging.DEBUG


class LastPartFilter(logging.Filter):
    def filter(self, record):
        record.name_last = record.name.rsplit('.', 1)[-1]
        return True


def reset_logger(base_level=BASE_LEVEL):
    logger = logging.getLogger(BASE_NAME)
    logger.setLevel(base_level)
    logger.handlers = []


def add_console_logger(level=logging.INFO):
    logger = logging.getLogger(BASE_NAME)
    ch = logging.StreamHandler(sys.stdout)
    ch.set_name('UPCP Console Logger')
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
    logger = logging.getLogger(BASE_NAME)
    fh = logging.FileHandler(log_path)
    fh.set_name('UPCP File Logger')
    fh.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    fh.addFilter(LastPartFilter())
    logger.addHandler(fh)


def set_console_level(level=logging.INFO):
    logger = logging.getLogger(BASE_NAME)
    for hl in logger.handlers:
        if hl.get_name() == 'UPCP Console Logger':
            hl.setLevel(level)
