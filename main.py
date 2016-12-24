import housing_price.housing_price as hp
import yaml

hp.collect()
with open('config.yml', 'r') as f:
    config = yaml.load(f)
