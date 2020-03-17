import logging
import os
import sys
import datetime
import re


def config_logging(log_dir, filename='log.txt'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    filepath = os.path.join(log_dir, filename)
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filepath, 'w')
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    logger.addHandler(ch)


def get_unique_dir(log_dir='', max_num=100, keep_original=False):
    if keep_original and not os.path.exists(log_dir):
        if log_dir == '':
            raise ValueError('log_dir cannot be empty with keep_original=True.')
        return log_dir
    else:
        for i in range(max_num):
            _dir = '{}-{}'.format(log_dir, i)
            if not os.path.exists(_dir):
                return _dir
        raise ValueError('Too many dirs starting with {}.'.format(log_dir))


def get_datetime():
    now = datetime.datetime.now().isoformat()
    now = re.sub(r'\D', '', now)[:-6]
    return now