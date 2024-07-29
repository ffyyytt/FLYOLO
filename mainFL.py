import os
import gc
import copy
import glob
import torch
import random
import argparse
import ultralytics

import flwr as fl
import numpy as np
import pandas as pd

from strategies import *
from sklearn.metrics import mean_squared_error
os.environ['YOLO_VERBOSE'] = 'False'

parser = argparse.ArgumentParser("YOLO")
parser.add_argument('-clients', help='YAML file of clients', nargs='+', required=True)
parser.add_argument("-group", help="Grouped dataset", nargs='?', default=None)
parser.add_argument("-rounds", help="Number of rounds", nargs='?', type=int, default=10)
parser.add_argument("-epochs", help="Number of local epochs", nargs='?', type=int, default=10)
parser.add_argument("-imgsz", help="Image size", nargs='?', type=int, default=512)
parser.add_argument("-batch", help="Batch size", nargs='?', type=int, default=16)
parser.add_argument("-seed", help="Random seed", nargs='?', type=int, default=1312)
parser.add_argument("-model", help="Model backbone", nargs='?', type=str, default="yolov8n.pt")
parser.add_argument("-strategy", help="Strategy Eg: FedAvg", nargs='?', type=str, default="FedAvg")
args = parser.parse_args()


args._clients = [x.replace(".yaml", "") for x in args.clients]
global_performance = {dname: [] for dname in args._clients + [args.group]}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(args.seed)
np.random.seed(args.seed)

if os.path.isdir(f"{args.strategy}_Output/{args._clients[0]}/"):
    print("Please remove Output before run a new experiment")
    exit(0)

def FLget_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in list(net.state_dict().items())]

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, file, trainloader, valloader, localEpochs):
        self.cid = cid
        self.trainloader = trainloader
        self.valloader = valloader
        self.localEpochs = localEpochs
        self.file = file

    def set_parameters(self, parameters):
        if len(glob.glob(f"{args.strategy}_Output/{args._clients[int(self.cid)]}/train*/weights/{self.file}.pt")) > 0:
            print(f"[Client {self.cid}] SET PARAMS {sorted(glob.glob(f'{args.strategy}_Output/{args._clients[int(self.cid)]}/train*/weights/{self.file}.pt'))[-1]}")
            model = torch.load(sorted(glob.glob(f"{args.strategy}_Output/{args._clients[int(self.cid)]}/train*/weights/{self.file}.pt"))[-1], map_location='cpu')
            for i, k in enumerate(model["model"].state_dict().keys()):
                try:
                    if len(model["model"].state_dict()[k].shape) > 1:
                        model["model"].state_dict()[k][:] = torch.from_numpy(parameters[i]).to(model["model"].state_dict()[k]).type(model["model"].state_dict()[k].type())
                    else:
                        model["model"].state_dict()[k] = torch.from_numpy(parameters[i]).to(model["model"].state_dict()[k]).type(model["model"].state_dict()[k].type())
                except:
                    pass
            torch.save(model, sorted(glob.glob(f"{args.strategy}_Output/{args._clients[int(self.cid)]}/train*/weights/{self.file}.pt"))[-1].replace(f"{self.file}.pt", "next.pt"))
            return ultralytics.YOLO(sorted(glob.glob(f"{args.strategy}_Output/{args._clients[int(self.cid)]}/train*/weights/next.pt"))[-1], verbose=False)
        print(f"[Client {self.cid}] SET PARAMS DEFAULT")
        return ultralytics.YOLO(args.model, verbose=False)

    def get_parameters(self, config):
        print(f"[Client {self.cid}] GET PARAMS {sorted(glob.glob(f'{args.strategy}_Output/{args._clients[int(self.cid)]}/train*/weights/{self.file}.pt'))[-1]}")
        model = torch.load(sorted(glob.glob(f"{args.strategy}_Output/{args._clients[int(self.cid)]}/train*/weights/{self.file}.pt"))[-1], map_location='cpu')
        return [val.cpu().numpy() for _, val in list(model["model"].state_dict().items())]

    def fit(self, parameters, config):
        gc.collect()
        model = self.set_parameters(parameters)
        metrics = self.__FLtrainModelWithModel(model, self.trainloader , epochs=self.localEpochs, config=config)
        return self.get_parameters({}), len(self.trainloader), metrics

    def evaluate(self, parameters, config):
        model = self.set_parameters(parameters)
        metrics = self.__FLmodelPredict(model, self.valloader)
        print(f"[Client {self.cid}] mAP@50: {float(metrics.box.map50)}")
        return 1-float(metrics.box.map50), len(self.valloader), {"mAP@50": float(metrics.box.map50)}
    
    def __FLtrainModelWithModel(self, model, dataLoader, epochs, config):
        if "proximal_mu" in config:
            optimizer = "ProxSGD"
        else:
            optimizer = "SGD"
        metrics = model.train(data = dataLoader,
                    epochs = epochs,
                    imgsz = args.imgsz,
                    seed = random.randrange(2**16),
                    batch = args.batch,
                    project= f"{args.strategy}_Output/{args._clients[int(self.cid)]}/",
                    optimizer = optimizer,
                    workers = 8)
        if optimizer == "ProxSGD":
            return metrics.results_dict | metrics.optimizer
        else:
            return metrics.results_dict

    def __FLmodelPredict(self, model, dataLoader):
        _model = copy.deepcopy(model)
        metrics = _model.val(data = dataLoader,
                             split = "val",
                             batch = args.batch)
        return metrics
    

def client_fn(cid):
    trainLoader = args.clients[int(cid)]
    validLoader = args.clients[int(cid)]
    return FlowerClient(cid, "last", trainLoader, validLoader, args.epochs).to_client()

client_resources = None
if device.type == "cuda":
    client_resources = {"num_cpus": 8, "num_gpus": 1}

if args.strategy in globals():
    strategy_fn = globals()[args.strategy]
else:
    raise "Strategy not exists"

strategy = strategy_fn(
    fraction_fit=1.,
    fraction_evaluate=1.,
    min_fit_clients=len(args._clients),
    min_evaluate_clients=len(args._clients),
    min_available_clients=len(args._clients),
    initial_parameters=fl.common.ndarrays_to_parameters(FLget_parameters(ultralytics.YOLO(args.model, verbose=False))),
)

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=len(args._clients),
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
    client_resources=client_resources,
)

for d in args._clients:
    dfs = []
    for round in range(args.rounds):
        dfs.append(pd.read_csv(f"{args.strategy}_Output/{d}/train{(round > 0)*str(round+1)}/results.csv"))
    df = pd.concat(dfs)
    df.to_csv(f"{args.strategy}_Output/{d.replace('/', '_')}.csv", index = False)

if not os.path.isdir(f"{args.strategy}_Output/{args.group}/"):
    os.makedirs(f"{args.strategy}_Output/{args.group}")
pd.DataFrame.from_dict(global_performance).to_csv(f"{args.strategy}_Output/{args.group}/global_performance.csv", index = False)