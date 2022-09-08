from distutils.command.config import config
import json

config_file = open("config.json","rb")
configs = json.load(config_file)

dataset_cfg=configs["dataset"]
data_attributes_cfg=configs["data attributes"]
testing_cfg=configs["testing dataset"]
train_cfg=configs["training"]
