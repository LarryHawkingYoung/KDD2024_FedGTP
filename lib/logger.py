import os
import logging
from datetime import datetime

def get_logger(args, root, name=None, debug=True):
    #when debug is true, show DEBUG and INFO in screen
    #when debug is false, show DEBUG in file and info in both screen&file
    #INFO will always be in screen
    # create a logger
    logger = logging.getLogger(name)
    #critical > error > warning > info > debug > notset
    logger.setLevel(logging.DEBUG)

    # define the formate
    formatter = logging.Formatter('%(asctime)s: %(message)s', "%Y-%m-%d %H:%M")
    # create another handler for output log to console
    console_handler = logging.StreamHandler()
    if debug:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)
        # create a handler for write log to file
        root = os.path.join(root, f"{args.dataset}_{args.exp_mode}_{args.cid}")
        if args.exp_mode in ['CTR', 'SGL']:
            file_name = f"{args.num_clients}p_{args.divide}_in{args.lag}_out{args.horizon}_{args.active_mode}.log"
        elif args.exp_mode == 'FED':
            if args.active_mode == 'sprtrelu':
                file_name = f"{args.num_clients}p_{args.divide}_in{args.lag}_out{args.horizon}_{args.active_mode}_{args.local_epochs}locals_{'fedavg' if args.fedavg else 'nofedavg'}.log"
            elif args.active_mode == 'adptpolu':
                file_name = f"{args.num_clients}p_{args.divide}_in{args.lag}_out{args.horizon}_{args.active_mode}_{args.act_k}th_{args.local_epochs}locals_{'fedavg' if args.fedavg else 'nofedavg'}.log"

        logfile = os.path.join(root, file_name)

        print('Creat Log File in: ', logfile)
        os.makedirs(root, exist_ok=True)
        file_handler = logging.FileHandler(logfile, mode='a+')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    # add Handler to logger
    logger.addHandler(console_handler)
    if not debug:
        logger.addHandler(file_handler)
    return logger


if __name__ == '__main__':
    time = datetime.now().strftime('%Y%m%d%H%M%S')
    print(time)
    logger = get_logger('./log.txt', debug=True)
    logger.debug('this is a {} debug message'.format(1))
    logger.info('this is an info message')
    logger.debug('this is a debug message')
    logger.info('this is an info message')
    logger.debug('this is a debug message')
    logger.info('this is an info message')