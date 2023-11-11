import os
import sys
import yaml
from logger import init_logger
from data_handlers.GCNdata import GCNData

import warnings
warnings.filterwarnings("ignore")

def main() -> None:
    logger = init_logger()
    with open('config.yaml') as file:
        params = yaml.safe_load(file)
    params['logger'] = logger
    data = GCNData(params)

if __name__=='__main__':
    main()