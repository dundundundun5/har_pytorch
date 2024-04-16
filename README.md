# 代码使用指南
1. cmd_generator.py 用于生成运行的命令行参数
2. get_sensor_from_dsads.py 用于小论文introduction出图
3. 其他暂时不写

## 运行需要
1. 环境名称 gjw 
2. 路径定位在 /data/dundun/har_pytorch


## 运行一条龙
1. cd /data/dundun/har_pytorch
2. conda activate gjw
<!-- 3. bash ./scripts/preprocess.sh  已经运行过数据预处理则无需再次运行-->
4. 打开cmd_generator.py, 直接运行，生成三个backbone的命令行参数, CUDA选3，用的人少
5. 保持路径不变 bash ./scripts/cuda3.sh 如果选了CUDA 1 则bash ./scripts/cuda1.sh
6. 如果只想要运行一条，则从cudaX.sh 取出命令行单独运行即可