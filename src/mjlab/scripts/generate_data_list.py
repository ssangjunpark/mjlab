from pathlib import Path

import yaml


def main():
  in_folder_path = "/home/sangjunpark/Documents/mjlab/unitree_g1_mjlab"
  out_path = "/home/sangjunpark/Documents/mjlab/unitree_g1_mjlab.yaml"
  files = list(Path(in_folder_path).rglob("*.npz"))
  d = []

  with open(out_path, "w") as f:
    for file in files:
      d.append(str(file))

    yaml.dump({"data": d}, f, default_flow_style=False, sort_keys=False)

  f.close()


def read():
  with open("/home/sangjunpark/Documents/mjlab/unitree_g1_mjlab.yaml") as stream:
    f = yaml.safe_load(stream)
    print(f["data"])


if __name__ == "__main__":
  # main()
  read()
