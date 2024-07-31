import json
from collections import OrderedDict


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
