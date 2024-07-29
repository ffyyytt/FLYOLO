import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.size'] = '16'

parser = argparse.ArgumentParser("YOLO")
parser.add_argument("-dataset", help="Dataset", nargs='?', type=str, default="ship_v0")
parser.add_argument("-group", help="Group Dataset", nargs='?', type=str, default="group4")
args = parser.parse_args()

dataset = args.dataset
group = args.group
results = {}

fedOutputs = glob("*_Output/")
for fedOutput in fedOutputs:
    csvfiles = glob(f"{fedOutput}{dataset}_*.csv")
    if len(csvfiles) > 0:
        results[fedOutput.replace("_Output/", "")] = np.max([pd.read_csv(csvfiles[i])['       metrics/mAP50(B)'] for i in range(len(csvfiles))], axis = 0)

if len(glob(f"Output/{dataset}/*/train/results.csv")) > 0:
    results["local"] = np.mean([pd.read_csv(f)["       metrics/mAP50(B)"][:200] for f in glob(f"Output/{dataset}/*/train/results.csv") if f != f"Output/{dataset}/{dataset}/train/results.csv"], axis = 0)
if os.path.isfile(f"Output/{dataset}/{group}/train/results.csv"):
    results["global"] = pd.read_csv(f"Output/{dataset}/{group}/train/results.csv")["       metrics/mAP50(B)"].values

for k, v in results.items():
    print(k, len(v))
    plt.plot(v[:200])

plt.xlabel('Epoch')
plt.ylabel('mAP')
plt.legend(results.keys(), loc=4)
plt.tight_layout()
plt.savefig(f'{dataset}.svg', format='svg')