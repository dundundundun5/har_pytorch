import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cycler
import matplotlib
print(matplotlib.matplotlib_fname())
plt.rc('font',family='Times New Roman')
LABEL = 12 # 12 or 18
LABEL = f"0{LABEL}" if LABEL < 10 else f"{LABEL}" 
NUM_P = 8
path = "../har_data/DSADS"
import os
os.makedirs("./dsads_fig", exist_ok=True)
# 橙色 绿色 蓝色
myCycler=cycler(color=['#FA8415', '#3BB44A', '#3C5DA8'])
plt.figure(figsize=(5,4))
plt.subplots_adjust(hspace=0.08,wspace=0.05)
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 8,
}
# 1 8 7 2
for i,p in enumerate([1,8,7,2]):
    
    # y = pd.read_csv(f"{path}/a{LABEL}/p{p}/s01.txt", header=None)
    y = np.loadtxt(f"{path}/a{LABEL}/p{p}/s01.txt", delimiter=",")

    x = np.arange(1, 126)
    y = y[:,[0,1,2]] / 1
    
    
    
    plt.subplot(4, 1, i+1)
    # plt.title(f"{i+1}")
    plt.gca().set_prop_cycle(myCycler)
    plt.xticks([])
    plt.xlim([0, 125])
    plt.yticks([])
    
    plt.plot(x, y,linewidth=1.2)
    # if i==3:
    #     plt.legend(["x","y","z"], loc="upper right", prop=font1)
    
plt.savefig(f"./dsads_fig/dsads_acc.png", dpi=600,bbox_inches='tight')

# plt.savefig(f"./dsads_fig/dsads_acc.svg", dpi=600,bbox_inches='tight')