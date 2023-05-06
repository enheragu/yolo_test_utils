
import yaml
from yaml.loader import SafeLoader

already_run_yaml = "test_cache_run.yaml"

with open(already_run_yaml) as file:
    already_run = yaml.load(file, Loader=SafeLoader)

with open(already_run_yaml, "w+") as file:
    yaml.dump(list(set(already_run)), file) # cast to set to remove duplicates