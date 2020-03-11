import logging
import os
import sys

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

def get_unique_dir(log_dir, max_num=100):
    if not os.path.exists(log_dir):
        return log_dir
    else:
        for i in range(max_num):
            _dir = '{}-{}'.format(log_dir, i)
            if not os.path.exists(_dir):
                return _dir
        raise ValueError('Too many dirs starting with {}.'.format(log_dir))