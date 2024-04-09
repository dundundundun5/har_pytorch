import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cycler

plt.rc('font',family='Times New Roman')
LABEL = 12 # 12 or 18
LABEL = f"0{LABEL}" if LABEL < 10 else f"{LABEL}" 
NUM_P = 8
path = "../har_data/DSADS"
import os
os.makedirs("./dsads_fig", exist_ok=True)
myCycler=cycler(color=['#7ac7e2', '#54beaa', '#e3716e'])
plt.figure(figsize=(5,4))
plt.subplots_adjust(hspace=0.08,wspace=0.05)
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 24,
}
# 1 8 7 2
for i,p in enumerate([1, 2, 3]):
    
    # y = pd.read_csv(f"{path}/a{LABEL}/p{p}/s01.txt", header=None)
    y = np.loadtxt(f"{path}/a{LABEL}/p{p}/s01.txt", delimiter=",")

    x = np.arange(1, 126)
    y = y[:,[0,1,2]] / 1.2
    # print(y[:,[0,1,2]].shape)
    
    
    plt.subplot(3, 1, i+1)
    # plt.title(f"{i+1}")
    plt.gca().set_prop_cycle(myCycler)
    plt.xticks([])
    
    plt.ylim([-5, 30])
    plt.yticks([])
    if i == 2:
        plt.xlabel("Time step(s)",fontdict=font1)
    # if i == 1:
    #     plt.ylabel("Accelerometer")
    
    plt.plot(x, y,linewidth=1.7)
    # if i==1:
    #     plt.legend(["acc_x","acc_y","acc_z"], loc=1, prop=font1)
    
plt.savefig(f"./dsads_fig/dsads.png", dpi=300,bbox_inches='tight')

plt.savefig(f"./dsads_fig/dsads.svg", dpi=300,bbox_inches='tight')