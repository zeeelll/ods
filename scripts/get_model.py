"""Download a YOLO weights file from Hugging Face and check it before use.

  python scripts/get_model.py REPO_ID [--file best.pt] [--out weights/drone.pt]

Prints the model's class names so you can confirm it has a drone class and set
the mapping in models.yaml. Check the model card's licence and training data
(RGB vs thermal!) before relying on it.
"""
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id")
    ap.add_argument("--file", default="best.pt")
    ap.add_argument("--out", default="weights/drone.pt")
    a = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    src = hf_hub_download(a.repo_id, a.file)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, out)
    names = YOLO(str(out)).names
    print(f"saved {out}\nclasses: {names}")
    if not any(k in n.lower() for n in names.values() for k in ("drone", "uav")):
        print("WARNING: no drone/uav class found - this model will not detect drones.")
    else:
        print("Set the right-hand side of `classes: {drone: ...}` in models.yaml to the matching name above.")


if __name__ == "__main__":
    main()
