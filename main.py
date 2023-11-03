import os
import sys
import yaml

class Config:
    def __init__(self, yamlFile):
        with open(yamlFile) as file:
            cfg = yaml.safe_load(file)
        self.params = cfgs