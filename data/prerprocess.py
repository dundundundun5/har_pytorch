import numpy as np
import pandas as pd
import os
from window import get_sliding_window
from scipy.io import loadmat
RELATIVE_PATH = "../har_data/" # with slash at the end
FINAL_PATH = RELATIVE_PATH + "preprocessed/" # 
DATA_PATH = {
    "dsads":"DSADS/",
    "mhealth":"MHEALTHDATASET/",
    "motionsense":"A_DeviceMotion_data/A_DeviceMotion_data/",
    "opportunity":"OpportunityUCIDataset/dataset/",
    "pamap2":"PAMAP2/Protocol/",
    "sho":"DataSet/",
    "uci_har":"UCI_HAR/",
    "unimib_shar":"UniMiB-SHAR/data/",
    "usc_had":"USC-HAD/",
}
LABEL_MAP = {
    "dsads":{1:0,2:1,3:2,4:3,5:4,6:5,7:6,8:7,9:8,10:9,11:10,12:11,13:12,14:13,15:14,16:15,17:16,18:17,19:18},
    "mhealth":{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12},
    "motionsense":{"dws":0,"jog":1, "sit":2,"std":3,"ups":4,"wlk":5},
    "opportunity":{406516:1,406517:2,404516:3,404517:4,
                   406520:5,404520:6,406505:7,404505:8,
                   406519:9,404519:10,406511:11,404511:12,
                   406508:13,404508:14,408512:15,407521:16,
                   405506:17,0:0}, # gestures 
                  # {4:3,5:4} # locomotion
    "pamap2":{24:12,19:0,20:0,0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,9:0,10:0,11:0,12:8,13:9,16:10,17:11,18:0},# {12:8,13:9,16:10,17:11,24:12,0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7}
    "sho":{"walking":0,"standing":1,"jogging":2,"sitting":3,"biking":4,"upstairs":5,"upsatirs":5,"downstairs":6},
    "uci_har":{1:0,2:1,3:2,4:3,5:4,6:5},
    "unimib_shar": {1:0,2:1},
    # two_classes {1:0,2:1}
    # adl {1:0,2:1,3:2,4:3,5:4,6:5,7:6,8:7,9:8}
    # fall {1:0,2:1,3:2,4:3,5:4,6:5,7:6,8:7}
    "usc_had":{1:0,2:1,3:2,4:3,5:4,6:5,7:6,8:7,9:8,10:9,11:10,12:11},
}
def normalize(x):
    # from sklearn.preprocessing import StandardScaler
    # s = StandardScaler()
    # x = s.fit_transform(x)
    
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    x = (x - mean) / (std + 0.0001)
    return x
def select_columns(data, dataset):
    if dataset == "opportunity":
        features_delete = np.arange(46, 50)
        features_delete = np.concatenate([features_delete, np.arange(59, 63)])
        features_delete = np.concatenate([features_delete, np.arange(72, 76)])
        features_delete = np.concatenate([features_delete, np.arange(85, 89)])
        features_delete = np.concatenate([features_delete, np.arange(98, 102)])
        features_delete = np.concatenate([features_delete, np.arange(134, 243)])
        features_delete = np.concatenate([features_delete, np.arange(244, 249)])
        data = np.delete(data, features_delete, 1)
    return data
def get_data_path(dataset:str):
    return RELATIVE_PATH + DATA_PATH[dataset], FINAL_PATH + dataset + "/"
def adjust_idx_labels(data_y:np.ndarray, dataset):
    return pd.Series(data_y).map(LABEL_MAP[dataset]).to_numpy()   
def divide_x_y(data, dataset):
    if dataset == "motionsense":
        pass
    elif dataset  == "dsads":
        data_x = data[:, :-1]
        data_y = data[:, -1] 
    elif dataset == "mhealth":
        data_x = data[:, :-1]
        data_y = data[:, -1] 
    elif dataset == "opportunity":
        data_x = data[:, 1:114]
        data_y = data[:, 115].astype(int)  # gestures
        # data_y = data_y[:, 114] # locomotion
    elif dataset == "pamap2":
        # 3~19 20~36 37~53 -> 4~15 21~32 38~49
        data_x = data[:, [4,5,6,7,8,9,10,11,12,13,14,15,21,22,23,24,25,26,27,28,29,30,31,32,38,39,40,41,42,43,44,45,46,47,48,49]]
        data_y = data[:, 1]
    elif dataset == "sho":
        data_x = data[:, :-1]
        data_y = data[:, -1]
    elif dataset == "unimib_shar":
        data_x = data[:, :-1]
        data_y = data[:, -1]
    elif dataset == "uci_har":
        raise NotImplemented("not for uci_har")
    elif dataset == "usc_had":
        data_x = data[:, :-1]
        data_y = data[:, -1]
    return data_x, data_y
def preprocess(data, dataset):
    data = select_columns(data, dataset)
    data_x, data_y = divide_x_y(data, dataset)
    data_y = adjust_idx_labels(data_y, dataset)
    data_y = data_y.astype(np.int64)
    data_x = data_x.astype(np.float64)
    np.nan_to_num(data_x, copy=False, nan=0)
    data_x[np.isnan(data_x)] = 0
    data_x = normalize(data_x)
    return data_x, data_y
# TODO dsads不再沿用师兄分割方法，改为follow官方分割
def dsads(sliding_window_length, sliding_window_step, path, final_path):
    for p in range(1, 8 + 1):
        res_x = np.empty((0, sliding_window_length , 45))
        res_y = np.empty((0, 1))
        for a in range(1, 19 + 1):
            
            data_x = np.empty((0, 45))
            data_y = np.empty((0))
            for s in range(1, 60 + 1):
                if s < 10:
                    s = "0" + str(s)
                data = np.load(f"/data/hzx/DSADS_npy/{p}-{a}-{s}.npy")
                data_x = data[: , :45]
                data_y = np.array([a-1 for _ in range(125)])
                
                mean = np.mean(data_x, axis=0)
                std = np.std(data_x, axis=0)
                data_x = (data_x - mean) / std
                
                temp_x, temp_y = get_sliding_window(data_x, data_y, sliding_window_length, sliding_window_step)
                res_x = np.concatenate([res_x, temp_x], axis=0)
                res_y = np.concatenate([res_y, temp_y.reshape(-1, 1)], axis=0)
        np.save(final_path + f"volunteer{p}_x.npy", res_x)
        np.save(final_path + f"volunteer{p}_y.npy", res_y)
        print(f"saving volunteer{p}_x.npy and volunteer{p}_y.npy")
# def dsads(sliding_window_length, sliding_window_step, path, final_path):
#     NUM_P = 9
#     NUM_A = 20
#     NUM_S = 61
#     NUM_CHANNELS = 46
#     for p in range(1, NUM_P):
#         res_x = np.empty((0, sliding_window_length , NUM_CHANNELS - 1))
#         res_y = np.empty((0, 1))
#         for a in range(1, NUM_A):
#             idx = a
#             if a < 10:
#                 a = "0" + str(a)
#             for s in range(1, NUM_S):
#                 if s < 10:
#                     s = "0" + str(s)
#                 data = pd.read_csv(path + f"a{a}/p{p}/s{s}.txt", header=None)
#                 x = data.to_numpy()
#                 y = np.array([idx-1])
#                 res_x = np.concatenate([res_x, x.reshape(1, 125, -1)], axis=0)
#                 res_y = np.concatenate([res_y, y.reshape(1, -1)], axis=0)
                
#         np.save(final_path + f"volunteer{p}_x.npy", res_x)
#         np.save(final_path + f"volunteer{p}_y.npy", res_y)
#         print(f"saving volunteer{p}_x.npy and volunteer{p}_y.npy")
def pamap2(sliding_window_length, sliding_window_step, path, final_path):
    NUM_P = 9
    for i in range(NUM_P):
        data = pd.read_csv(path + f"subject10{i + 1}.dat", sep=' ', header=None)
        data = data.to_numpy()
        x, y = preprocess(data, "pamap2")
        x, y = get_sliding_window(x, y, sliding_window_length, sliding_window_step)
        #classes(y)
        np.save(final_path + f"volunteer{i + 1}_x.npy", x)
        np.save(final_path + f"volunteer{i + 1}_y.npy", y)
        print(f"保存文件volunteer{i + 1}_x.npy和volunteer{i + 1}_y.npy")
def opportunity(sliding_window_length, sliding_window_step, path, final_path):
    NUM_CHANNELS = 114
    NUM_P = 4
    # TODO oppotunity填充nan值方法无效 解决
    for i in range(NUM_P):
        data_x = np.empty((0, sliding_window_length , NUM_CHANNELS - 1))
        data_y = np.empty((0, 1),dtype=int)
        for j in range(5):
            data = pd.read_csv(path + f"S{i + 1}-ADL{j + 1}.dat", sep=" ", header=None).to_numpy()
            x, y = preprocess(data,"opportunity")
            x, y = get_sliding_window(x, y, sliding_window_length, sliding_window_step)
            data_x = np.concatenate([data_x, x])
            data_y = np.concatenate([data_y, y])
        data = pd.read_csv(path + f"S{i + 1}-Drill.dat", sep=" ", header=None).to_numpy()
        x, y = preprocess(data,"opportunity")
        x, y = get_sliding_window(x, y, sliding_window_length, sliding_window_step)
        data_x = np.concatenate([data_x, x])
        data_y = np.concatenate([data_y, y])
        #classes(data_y)
        np.save(file=final_path + f"volunteer{i + 1}_x.npy", arr=data_x)
        np.save(file=final_path + f"volunteer{i + 1}_y.npy", arr=data_y)
        print(f"保存文件volunteer{i + 1}_x.npy和volunteer{i + 1}_y.npy")
def usc_had(sliding_window_length, sliding_window_step, path, final_path):
    NUM_A = 13
    NUM_T = 6
    NUM_P = 15
    NUM_CHANNELS = 7
    for subject in range(1, NUM_P):
        data_x = np.zeros((0, sliding_window_length, NUM_CHANNELS - 1))
        data_y = np.zeros((0, 1))
        for a in range(1, NUM_A):
            for t in range(1, NUM_T):
                print(f"reading Subject{subject}/a{a}t{t}.mat")
                mat = loadmat(path + f"Subject{subject}/a{a}t{t}.mat")
                data = pd.DataFrame(mat['sensor_readings'])
                data['label'] = a
                data = data.to_numpy()
                x, y = preprocess(data, "usc_had")
                x, y = get_sliding_window(x, y, sliding_window_length, sliding_window_step)
                data_x = np.concatenate([data_x, x])
                data_y = np.concatenate([data_y, y])
                np.save(path + f"Subject{subject}/a{a}t{t}.npy", data)
        #classes(data_y)
        np.save(final_path + f"volunteer{subject}_x.npy", data_x)
        np.save(final_path + f"volunteer{subject}_y.npy", data_y)
        print(f"saving volunteer{subject}_x.npy and volunteer{subject}_y.npy")
def unimib_shar(sliding_window_length, sliding_window_step, path,final_path):
    NUM_P = 30 # 获取志愿者总人数   
    TYPE = "adl" # "adl" "fall"
    # 将各部分的数据按人划分为npy文件-----------------------------------------------
    data = loadmat(path + TYPE + "_data.mat")[TYPE + "_data"]
    labels = loadmat(path + TYPE + "_labels.mat")[TYPE + "_labels"]
    # 按人划分
    res = [np.empty((0,data.shape[1] + 1)) for _ in range(NUM_P)] # 0~max_subjects-1 and label
    for i in range(data.shape[0]):
        subject_id = labels[i][1] - 1 # 获取志愿者编号 1~max_subjects - 1 =  0~max_subjects-1
        temp_data = data[i].reshape(1, -1)
        temp_label = np.array([labels[i][0]], dtype=np.int64).reshape(-1, 1)
        temp = np.concatenate([temp_data, temp_label], axis=1)
        res[subject_id] = np.concatenate([res[subject_id], temp])
        # 第i个志愿者的动作标签
    for i in range(NUM_P):
        x, y = preprocess(res[i], "unimib_shar")
        x = x.reshape(-1, 151, 3)
        y = y.reshape(-1, 1)
        #classes(y)
        np.save(final_path + f"volunteer{i + 1}_x.npy", x)
        np.save(final_path + f"volunteer{i + 1}_y.npy", y)
        print(f"saving volunteer{i + 1}_x.npy and volunteer{i + 1}_y.npy")
def sho(sliding_window_length, sliding_window_step, path, final_path):
    NUM_P = 10
    for i in range(NUM_P):
        data = pd.read_csv(path + f"Participant_{i + 1}.csv", sep="[,]+",header=1, engine="python")
        c = data.columns.to_numpy()
        c[-1] = "label"
        data.columns = c
        data.drop(["time_stamp","time_stamp.1","time_stamp.2","time_stamp.3","time_stamp.4"], inplace=True, axis=1)
        data = data.to_numpy()
        x, y = preprocess(data, "sho")
        x, y = get_sliding_window(x, y, sliding_window_length, sliding_window_step)
        #classes(y)
        np.save(file=final_path + f"volunteer{i + 1}_x.npy", arr=x)
        np.save(file=final_path + f"volunteer{i + 1}_y.npy", arr=y)
        print(f"保存文件volunteer{i + 1}_x.npy和volunteer{i + 1}_y.npy")
def mhealth(sliding_window_length, sliding_window_step, path, final_path):
    NUM_P = 10
    for i in range(NUM_P):
        data = pd.read_csv(path + f"mHealth_subject{i + 1}.log", sep="\\s+", header=None).to_numpy()
        x, y = preprocess(data, "mhealth")
        x, y = get_sliding_window(x, y, sliding_window_length, sliding_window_step)
        #classes(y)
        np.save(file=final_path + f"volunteer{i + 1}_x.npy", arr=x)
        np.save(file=final_path + f"volunteer{i + 1}_y.npy", arr=y)
        print(f"保存文件volunteer{i + 1}_x.npy和volunteer{i + 1}_y.npy")
def motionsense(sliding_window_length, sliding_window_step, path, final_path):
    NUM_P = 24
    NUM_CHANNELS = 12
    types = {"dws":0,"jog":1,"sit":2,"std":3,"ups":4,"wlk":5}
    FILES = ["dws_1","dws_2","dws_11","jog_9","jog_16","sit_5",
             "sit_13","std_6","std_14","ups_3","ups_4","ups_12",
             "wlk_7","wlk_8","wlk_15"]
    for i in range(NUM_P):
        data_x = np.empty((0, sliding_window_length, NUM_CHANNELS))
        data_y = np.empty((0, 1),dtype=int)
        for file in FILES:
            data = pd.read_csv(path + f"{file}/sub_{i + 1}.csv")
            label = types[file[:3]]
            data = data.to_numpy()
            x = data[:, 1:]
            y = np.zeros((x.shape[0],))
            y.fill(label)
            x[np.isnan(x)] = 0
            x = normalize(x)
            x, y = get_sliding_window(x, y, sliding_window_length, sliding_window_step)
            data_x = np.concatenate([data_x, x])
            data_y = np.concatenate([data_y, y])
        #classes(data_y)
        np.save(file=final_path + f"volunteer{i + 1}_x.npy", arr=data_x)
        np.save(file=final_path + f"volunteer{i + 1}_y.npy", arr=data_y)
        print(f"保存文件volunteer{i + 1}_x.npy和volunteer{i + 1}_y.npy")
def uci_har(sliding_window_length, sliding_window_step, path,final_path):
    INPUT_SIGNAL_TYPES = [
            "body_acc_x",
            "body_acc_y",
            "body_acc_z",
            "body_gyro_x",
            "body_gyro_y",
            "body_gyro_z",
            "total_acc_x",
            "total_acc_y",
            "total_acc_z"
        ]
    # RUN JUST ONCE 
    for file, n in {"train":7352, "test":2947}.items():
        total_x = np.zeros((n, 128, 0))
        for type in INPUT_SIGNAL_TYPES:
            x = np.loadtxt(f"{path}{file}/Inertial Signals/{type}_{file}.txt", np.float32)
            total_x = np.concatenate([total_x, x.reshape(n, 128, 1)],axis=2)
        np.save(f"{path}{file}/{file}_X.npy", total_x)
    # RUN JUST ONCE
    train_X = np.load(path + "train/train_X.npy") 
    train_y = pd.read_csv(path + "train/" + "y_train.txt", sep=' ', header=None).to_numpy().reshape(-1)
    train_p = pd.read_csv(path + "train/" + "subject_train.txt", sep=' ', header=None).to_numpy().reshape(-1)
    test_X = np.load(path + "test/test_X.npy")
    test_y = pd.read_csv(path + "test/" + "y_test.txt", sep=' ', header=None).to_numpy().reshape(-1)
    test_p = pd.read_csv(path + "test/" + "subject_test.txt", sep=" ", header=None).to_numpy().reshape(-1)
    # activity ID range start from 0
    train_y, test_y = adjust_idx_labels(train_y,"uci_har"), adjust_idx_labels(test_y,"uci_har")
    data_x = [np.empty((0, 128,train_X.shape[2])) for i in range(30)]
    data_y = [np.empty((0,1),dtype=int) for i in range(30)]
    # # 6类 30个志愿者
    for i in range(len(train_X)):
        index = train_p[i]
        data_x[index - 1] = np.concatenate([data_x[index - 1], train_X[i].reshape(1, 128, 9)], axis=0)
        data_y[index - 1] = np.concatenate([data_y[index - 1], train_y[i].reshape(-1,1)])
    for i in range(len(test_X)):
        index = test_p[i]
        data_x[index - 1] = np.concatenate([data_x[index - 1], test_X[i].reshape(1, 128, 9)], axis=0)
        data_y[index - 1] = np.concatenate([data_y[index - 1], test_y[i].reshape(-1,1)])
    for i in range(30):
        np.save(final_path + f"volunteer{i + 1}_x.npy", data_x[i])
        np.save(final_path + f"volunteer{i + 1}_y.npy", data_y[i])
        #classes(data_y[i])
        print(f"saving volunteer{i + 1}_x.npy and volunteer{i + 1}_y.npy")
def main(sliding_window_length, sliding_window_step, dataset):
    
    path, final_path = get_data_path(dataset=dataset)
    if dataset == "dsads":
        final_path += "32/"
    elif dataset == "uci_har":
        final_path += "128/"
    elif dataset == "unimib_shar":
        final_path += "151/"
    else:
        final_path += f"{sliding_window_length}/"
    os.makedirs(final_path, exist_ok=True)
    eval(dataset)(sliding_window_length, sliding_window_step, path, final_path)
def classes(data_y):
    print(f"===============y{data_y.shape}==============")
    print(pd.DataFrame(data_y).value_counts().sort_index())
if __name__ == '__main__':
    
    main(64, 32, "opportunity")