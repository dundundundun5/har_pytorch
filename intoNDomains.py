import numpy as np
strategy = {"uci_har":[[1,2,3,4,5,6,7],[8,9,10,11,12,13,14],[15,16,17,18,19,20,21],[22,23,24,25,26,27,28]],'pamap2':[[1],[2],[3],[4],[5],[6],[7],[8]],'opportunity':[[1],[2],[3],[4]], 'unimib_shar':[[1,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20],[21,22,23,24,25,26,27,28,29,30]],'dsads':[[1],[2],[3],[4],[5],[6],[7],[8]]}
dataset_dict = eval(open("./data/dataset_dict.json", "r+").read())

for dataset,window_len in [('dsads',32),('uci_har', 128), ('pamap2',128), ('unimib_shar',151), ('opportunity',64)]:

    for i in range(len(strategy[dataset])):
        c = dataset_dict[dataset]['num_channels']
        res_x = np.empty((0, window_len, c))
        res_y = np.empty((0,1))
        
        for j in range(len(strategy[dataset][i])):
            
            idx = strategy[dataset][i][j]
            x = np.load(f"../har_data/preprocessed/{dataset}/{window_len}/volunteer{idx}_x.npy")
            
            y = np.load(f"../har_data/preprocessed/{dataset}/{window_len}/volunteer{idx}_y.npy")
            
            res_x = np.concatenate([res_x, x], axis=0)
            res_y = np.concatenate([res_y, y], axis=0)
        np.save(f"../har_data/preprocessed/{dataset}/{window_len}/domain{i+1}_x.npy", res_x)
        np.save(f"../har_data/preprocessed/{dataset}/{window_len}/domain{i+1}_y.npy", res_y)

        