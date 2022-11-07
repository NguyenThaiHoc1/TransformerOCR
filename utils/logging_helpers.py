import os
import pytz
import logging
import datetime
from logging.handlers import TimedRotatingFileHandler


def setting_logging(path_logging):
    path_logging = os.path.join(str(path_logging), '{}-{}-{}_{}.log')
    universal = datetime.datetime.utcnow()
    universal = universal.replace(tzinfo=pytz.timezone("Asia/Ho_Chi_Minh"))
    formatter = logging.Formatter("%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s")
    handler = TimedRotatingFileHandler(path_logging.format(universal.year,
                                                           universal.month,
                                                           universal.day,
                                                           universal.hour),
                                       when="midnight",
                                       interval=1, encoding="utf-8")

    handler.suffix = "%Y-%m-%d"
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    logger.addHandler(handler)
