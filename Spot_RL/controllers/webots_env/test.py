import yaml

with open('spot_env_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

print(config)
