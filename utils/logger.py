import logging
import os
import sys
from datetime import datetime, date


class Logger:

    @classmethod
    def init_logger(cls, log_path, log_name):
        if not os.path.isdir(log_path):
            os.makedirs(log_path)
        log_path = os.path.join(log_path, f'{str(date.today())}_{log_name}.log')
        logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO, filename=log_path)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))