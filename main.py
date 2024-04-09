from argparse import ArgumentParser
import torch
from utils import *

def main(args):
    setup_seed(args.seed)
    data_loaders = get_loader_by_volunteers(args)
    result_name, result_path, result_now_path = get_result_name(args)
    # end2end moe
    if args.use_bagging == 1:
        models = [get_model(args) for _ in range(args.n_domains - 1)]
        res = train_test_bagging_with_torchmetrics(models, data_loaders[0], data_loaders[1], result_name, result_path, args)
        exit(0)
    if args.use_moe == 1 and args.end2end == 1:
        args.test_p = args.volunteer_split[-1]
        models = [get_model(args) for _ in range(len(data_loaders) - 1)]
        from model.Gate import Gate3
        moe = Gate3(args.sliding_window_length, args.num_channels, len(data_loaders) - 1, args.num_hiddens)
        res = train_test_moe_end2end_with_torchmetrics(models, moe, data_loaders[:-1],data_loaders[-1],result_name, result_path, args)
        if args.return_loss == 1:
            print(f"all_loss={res[0]}")
            print(f"domain_loss={res[1]}")
            print(f"all_moe_loss={res[2]}")
            print(f"all_equal_loss={res[3]}")
            
            np.save(f"./scripts/{args.dataset_name}_all_loss.npy", np.array(res[0]))
            np.save(f"./scripts/{args.dataset_name}_all_domain_loss_{args.alpha}.npy",np.array(res[1]))
            np.save(f"./scripts/{args.dataset_name}_all_moe_loss_{args.alpha}.npy", np.array(res[2]))
            np.save(f"./scripts/{args.dataset_name}_all_equal_loss_{args.beta}.npy", np.array(res[3]))
            np.save(f"./scripts/{args.dataset_name}_all_acc.npy", np.array(res[4]))
            exit(0)
    # 2stage moe
    elif args.use_moe == 1 and args.end2end == 0:
        from model.Gate import Gate3
        moe = Gate3(args.sliding_window_length, args.num_channels, len(data_loaders) - 1, 256)
        res = train_test_moe_2stage_with_torchmetrics(moe, data_loaders[:-1], data_loaders[-1], result_name, result_path, args)
    # train single model
    elif args.use_moe == 0:
        model = get_model(args)
        if args.model_name == 'dann':
            res = train_test_dann_with_torchmetrics(model, data_loaders[0], data_loaders[1], result_name, result_path, args)
        elif args.model_name == 'deepcoral':    
            res = train_test_deepcoral_with_torchmetrics(model, data_loaders[0], data_loaders[1], result_name, result_path, args)
        elif args.model_name == 'rsc':
            res = train_test_rsc_with_torchmetrics(model, data_loaders[0], data_loaders[1], result_name, result_path, args)
        else:
            res = train_test_with_torchmetrics(model, data_loaders[0], data_loaders[1], result_name, result_path, args)
    # save all the results     
    save_by_result_name(result_path, res, args)
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--repeat', default=1, type=int)
    parser.add_argument('--seed', default=3407, type=int)
    # cuda device 0
    parser.add_argument('--cuda_device', default=2, type=int)
    parser.add_argument('--dataset_name', default='opportunity',
                        choices=['dsads', 'opportunity', 'pamap2',
                                 'uci_har', 'usc_had',
                                 'unimib_shar', "mhealth", "motionsense"],
                        type=str)
    parser.add_argument('--only_preprocess', default=0, type=int)
    parser.add_argument('--only_info', default=1, type=int)
    parser.add_argument('--sliding_window_length', default=32, type=int)
    parser.add_argument('--sliding_window_step', default=28, type=int)
    parser.add_argument('--volunteer_split', default="1,2,3,4,5,6,7|8", type=str)
    # train info
    parser.add_argument('--model_name', default='mtsdnet',
                        type=str, choices=['deepconvlstm','transformer','mtsdnet','ddnn',"tripleattention","dualattention","gile", "mtsdnet_layer_moe","dann", "deepcoral","rsc"])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--use_moe', type=int, default=0)
    parser.add_argument('--use_bagging', type=int, default=0)
    parser.add_argument('--end2end', type=int, default=0)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--lr', default=3e-3, type=float)
    parser.add_argument('--wd', default=0, type=float)
    parser.add_argument('--criterion', default='ce', type=str)
    # Model Hyperparameters 这里不急着添加
    parser.add_argument('--num_hiddens', default=4, type=int) # moe
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--beta', default=0.5, type=float)
    parser.add_argument('--equal', type=int, default=0)
    parser.add_argument('--return_loss', type=int, default=1)
    # GILE model parameters
    parser.add_argument("--n_domains", default=4, type=int)
    # DDNN model parameters
    parser.add_argument("--n_lstm_hidden", default=128, type=int)
    parser.add_argument('--d_AE', type=int, default=50, help='dim of AE')
    parser.add_argument("--out_channels", default=64, type=int)
    # Model Save and Load Path
    parser.add_argument('--save_path', default='../4domain_results')
    parser.add_argument('--now', default=get_time())
    parser.add_argument("--hidden", default=128, type=int)
    parser.add_argument("--dim", default=64, type=int)
    parser.add_argument("--structure_str", default="tsg", type=str)
    
    args = parser.parse_args()
    # 录入数据集指标
    dataset_dict = eval(open("./data/dataset_dict.json", "r+").read())
    args.num_classes = dataset_dict[args.dataset_name]['num_classes']
    args.num_channels = dataset_dict[args.dataset_name]['num_channels']
    args.num_volunteers = dataset_dict[args.dataset_name]['num_volunteers']
    if not torch.cuda.is_available():
        raise RuntimeError("no cuda!!!")
    # 统计模型参数
    if args.only_info == 1:
        get_model_info(args)
        exit(0)
    # 数据处理
    if args.only_preprocess == 1:
        dataset_preprocess(args)
        exit(0)
    for repeat_i in range(args.repeat):
        args.seed = 3407
        print(f"repeat#{repeat_i+1}===seed={args.seed}")
        main(args)

    