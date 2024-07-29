import torch
import argparse
import ultralytics

parser = argparse.ArgumentParser("YOLO")
parser.add_argument("-data", help="Dataset file", nargs='?', type=str, default="ship_v0/planet.yaml")
parser.add_argument("-epochs", help="Number of local epochs", nargs='?', type=int, default=100)
parser.add_argument("-imgsz", help="Image size", nargs='?', type=int, default=512)
parser.add_argument("-seed", help="Random seed", nargs='?', type=int, default=1312)
parser.add_argument("-batch", help="Batch size", nargs='?', type=int, default=16)
parser.add_argument("-model", help="Model backbone", nargs='?', type=str, default="yolov8n.pt")
parser.add_argument('-eval', help='YAML file of evaluates', nargs='+', default=[])
args = parser.parse_args()

model = ultralytics.YOLO(args.model, verbose=False)

H = model.train(data = f'{args.data}',
                epochs = args.epochs,
                imgsz = args.imgsz,
                seed = args.seed,
                batch = args.batch,
                project= f"Output/{args.data.replace('.yaml', '')}/",
                optimizer = "SGD",
                patience = 200,
                workers = 8)

model = ultralytics.YOLO(f"Output/{args.data.replace('.yaml', '')}/train/weights/best.pt")
for data in args.eval:
    print("Train: {args.data}, Load: best, Valid: {data}")
    model.val(data=data, split = "val")