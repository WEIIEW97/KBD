import json
from collections import OrderedDict

import yaml


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


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        content = json.load(f)
    return content


def write_to_json(content: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(content, f, indent=4)

    print(f"Data has been written to {path}")


if __name__ == "__main__":
    config_path = "../configs/KBD_config.json"

    TEMPLATE_CONFIG = {
        "save_dir": "D:/william/codes/KBD/data/N09ASH24DH0056",
        "root_dir": "D:/william/codes/KBD/data/N09ASH24DH0056/image_data",
        "result_dir": "D:/william/codes/KBD/data/N09ASH24DH0056/image_data_TRR",
        "table_path": "D:/william/codes/KBD/data/N09ASH24DH0056/depthquality-2024-05-22_56.xlsx",
    }

    write_to_json(TEMPLATE_CONFIG, config_path)

    content = load_json(config_path)
    print(content)
