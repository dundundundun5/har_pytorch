CUDA = "0" # 虚拟GPU号 0~3
META = "0" # 1 要 META
REPEAT = 1 # 实验重复次数
V = "../meta_results"
models = ['resnet',"tripleattention","dualattention","mtsdnet","ddnn","transformer",'gile', 'rsc', 'deepconvlstm','gruinc','tccsnet','resnetmeta']
for BASELINE in models:
    # datasets = [ ("dsads","125", "128", 8),("pamap2","128", "128", 8), ("opportunity", "64", "128", 4), ("uci_har","128","128", 6), ("unimib_shar", "151", "128", 8),("mhealth", "128", "128", 5), ("usc_had", "128", "128", 7), ("motionsense", "128", "128", 8)]
    datasets = [("pamap2","128", "128", 8), ("opportunity", "64", "128", 4), ("uci_har","128","128", 6), ("unimib_shar", "151", "128", 8),("mhealth", "128", "128", 5), ("usc_had", "128", "128", 7)]
    for i, dataset in enumerate(datasets):
        dataset_dict = eval(open("./data/dataset_dict.json", "r+").read())
        params =   {"dataset_name":"" ,  "model_name":"caonima" ,  "sliding_window_length": "" ,  "batch_size":"" ,  "epochs":"30" ,  "cuda_device":CUDA ,  "use_meta":META, "repeat": REPEAT, "beta":"0.5", "gamma":"0.5", "save_path":V} 
        params['model_name'] = BASELINE
        params["dataset_name"],  params['sliding_window_length'] , params['batch_size'], num_volunteers = dataset
        

        def split_generator(num_volunteers, use_meta):
            
            volunteer_splits = []
        # 1,2,3\|4
            use_meta = int(use_meta)
            if use_meta == 0:
                for i in range(1, num_volunteers+1):
                    
                    temp = f""
                    for j in range(1, num_volunteers+1):
                        
                        if j == i:
                            continue
                        temp += f"{j},"
                    temp = temp[:-1]
                    temp += f"\\|{i}"
                    volunteer_splits.append(temp)
            if use_meta == 1:
                for i in range(1, num_volunteers+1):
                    
                    temp = "\\|"
                    all_temp = ""
                    for j in range(1, num_volunteers+1):
                        
                        if j == i:
                            continue
                        temp += f"{j}\\|"
                        all_temp += f",{j}"
                    temp = temp[:-2]
                    all_temp = all_temp[1:]
                    temp += f"\\|{i}"
                    res = all_temp + temp
                    volunteer_splits.append(res)
                    
            return volunteer_splits

        results = []
        if params['model_name'] == 'resnetmeta':
            params["use_meta"] = 1
        else:
            params["use_meta"] = 0
        volunteer_splits = split_generator(num_volunteers, params["use_meta"])
        for volunteer_split in volunteer_splits:
            result = "python main.py "
            for key,value in params.items():
                result += f"--{key} {value} "
            result += f"--volunteer_split {volunteer_split} "
            
            result += f"--n_domains {num_volunteers} "
            results.append(result)
            print(result)
            

        with open(f"./scripts/meta_cuda{params['cuda_device']}.sh", mode="a+", encoding="utf-8") as f:
            
            print(f"{params['dataset_name']}_cuda{params['cuda_device']}")    
            for r in results:
                f.write(r + "\n\n")