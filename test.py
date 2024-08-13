import os
import yaml
import argparse
import ultralytics
from glob import glob

parser = argparse.ArgumentParser("YOLO")
parser.add_argument("-data", help="YAML file", nargs='?', type=str, default="ship_v0/planet.yaml")
parser.add_argument("-split", help="split: val, test, train", nargs='?', type=str, default="val")
parser.add_argument("-imgsz", help="Image size", nargs='?', type=int, default=512)
parser.add_argument("-batch", help="Batch size", nargs='?', type=int, default=16)
parser.add_argument("-model", help="Model weight", nargs='?', type=str, default="model.pt")
parser.add_argument("-output", help="Folder Output", nargs='?', type=str, default="predict")
args = parser.parse_args()

if not os.path.exists(args.output):
    os.makedirs(args.output)

model = ultralytics.YOLO(args.model, verbose=False)
model.val(data=args.data, 
          imgsz = args.imgsz,
          batch = args.batch,
          split = args.split)
path = yaml.safe_load(open(args.data))[args.split]
if isinstance(path, list):
    files = sum([glob(p) for p in path])
else:
    files = glob(path)

results = model(files)
for i, result in enumerate(results):
    result.save(filename=os.path.join(args.output, files[i].split("/")[-1]))