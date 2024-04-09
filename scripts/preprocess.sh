python ../dsads.py 

python main.py --only_preprocess 1 --dataset_name opportunity --sliding_window_length 64 --sliding_window_step 32

python main.py --only_preprocess 1 --dataset_name pamap2 --sliding_window_length 128 --sliding_window_step 32

python main.py --only_preprocess 1 --dataset_name uci_har --sliding_window_length 128 --sliding_window_step -1

python main.py --only_preprocess 1 --dataset_name unimib_shar --sliding_window_length 151 --sliding_window_step -1

python intoNDomains.py

