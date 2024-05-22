import yaml
from collections import OrderedDict


def ordered_dict_representer(dumper, data):
    return dumper.represent_dict(data.items())


yaml.add_representer(OrderedDict, ordered_dict_representer)


def create_default_KBD_config(path: str, content: dict):
    with open(path, "w") as f:
        yaml.dump(content, f, default_flow_style=None, sort_keys=False)


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        content = yaml.safe_load(f)

    return content


if __name__ == "__main__":
    config_path = "../configs/KBD_condig.yaml"

    TEMPLATE_CONFIG = {"H": 480, "W": 640, "epsilon": 1e-6, "anchor point": [240, 320]}

    create_default_KBD_config(config_path, TEMPLATE_CONFIG)

    content = load_yaml(config_path)
    print(content)
