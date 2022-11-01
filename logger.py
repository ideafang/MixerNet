import logging
# from x2 import do_sth

# logging.basicConfig(filename='train.log', level=logging.INFO, format='%(asctime)s - %(filename)s - %(message)s')
# logging.info("test logging")
# do_sth()
# logging.info("finish test")

def config_logger(mode='train', filename='train.log'):
    logger = logging.getLogger(mode)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    fh = logging.FileHandler(filename, 'w', encoding='utf-8')

    formatter = logging.Formatter('%(asctime)s - %(filename)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

# logger = config_logger(mode='train')
# logger.info("xxx info")
# do_sth()
# logger.info("finish info")
