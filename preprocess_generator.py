from argparse import ArgumentParser
dataset_dict = eval(open("./data/dataset_dict.json", "r+").read())
DATASETS = {
    "dsads": {
        "sliding_window_length": 32,
        "sliding_window_step": 28
    },
    "opportunity": {
        "sliding_window_length": 64,
        "sliding_window_step": 32
    },
    "pamap2": {
        "sliding_window_length": 128,
        "sliding_window_step": 64
    },
    "uci_har": { 
        "sliding_window_length": 128,
        "sliding_window_step": -1     
    },
    "usc_had": {
        "sliding_window_length": 128,
        "sliding_window_step": 64
    },
    "unimib_shar": {
        "sliding_window_length": 151,
        "sliding_window_step": -1
    },
    "mhealth": {
        "sliding_window_length": 128,
        "sliding_window_step": 64
    },
    "motionsense": {
        "sliding_window_length": 128,
        "sliding_window_step": 64
    },
}
ans = []
for dataset,value in DATASETS.items():
    res = 'python main.py --only_preprocess 1 '
    res += f"--dataset_name {dataset} "
    res += f"--sliding_window_length {value['sliding_window_length']} --sliding_window_step {value['sliding_window_step']}\n\n"
    ans.append(res)
import os
os.makedirs("./scripts/", exist_ok=True)    
with open("./scripts/preprocess.sh", "w") as f:
    for a in ans:
        f.write(a)
    f.write("python intoNDomains.py\n\n")