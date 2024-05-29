from KBD.apis import generate_parameters_trf, apply_transformation_parallel
from KBD.helpers import parallel_copy
import os
import json

CONFIG_PATH = "configs/KBD_config.json"

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        content = json.load(f)
    return content


def write_to_json(content: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(content, f, indent=4)

    print(f"Data has been written to {path}")


if __name__ == "__main__":
    cwd = os.getcwd()
    params = load_json(os.path.join(cwd, CONFIG_PATH))
    rootdir = params["root_dir"]
    savedir = params["save_dir"]
    table_path = params["table_path"]
    resultdir = params["result_dir"]

    print(f"processing {rootdir} now with {table_path} ...")
    k, delta, b, focal, baseline = generate_parameters_trf(
        path=rootdir,
        tabel_path=table_path,
        save_path=savedir,
    )

    parallel_copy(rootdir, resultdir)
    apply_transformation_parallel(resultdir, k, delta, b, focal, baseline)

    print("Working done ...")  
    os.system('pause')