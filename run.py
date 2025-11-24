import numpy as np
import torch
from data_loader import GBMGraphDataset
from torch_geometric.data import DataLoader
from utils import check_dir, setup_seed, get_logger
from train import train
import torch.nn as nn

import torch.optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts, CyclicLR, OneCycleLR, \
    CosineAnnealingLR, SequentialLR, ConstantLR, ExponentialLR
import argparse
import datetime
from torch.utils.tensorboard import SummaryWriter
import time
import json
import subprocess
from models import TwinsGCN



def params_setup():
    params = dict()
    params["in_chans"] = [768]
    params["out_chans"] = [16]
    params["heads"] = [8]

    return params


def logger_initial(logger, params):
    logger.info('params["in_chans"]=[%d]' % params["in_chans"][0])
    logger.info('params["out_chans"]=[%d]' % params["out_chans"][0])
    logger.info('params["heads"]=[%d]' % params["heads"][0])

    return logger


class savePath():
    def __init__(self, time_tab):
        
        partient_dir = "./TrainProcess/"

        self.model_path = partient_dir + time_tab + "/" + "model" + "/"
        check_dir(self.model_path)
    
        self.log_path = partient_dir + time_tab + "/" + time_tab + ".log"

        self.writer_path = partient_dir + time_tab + "/" + "tensorboard" + "/" + time_tab
        check_dir(self.writer_path)
        self.argument_path = partient_dir + time_tab + "/" + time_tab + ".json"


class TrainingConfig():
    def __init__(self, logger, writer, save_path, args):
        self.logger = logger
        self.writer = writer
        self.model_path = save_path.model_path
        self.num_epochs = args.num_epochs
        self.device = args.device
        self.scheduler = args.scheduler
        self.savedisk = args.savedisk
        self.smoothing = args.smoothing




def main(args, model, model_architecture, time_tab):
    
    # set path for saving
    save_path = savePath(time_tab)
    logger = get_logger(save_path.log_path)
    writer = SummaryWriter(save_path.writer_path)
    
    logger.info("LEARNING RATE: %s, OPTIMIZER: %s, NUM EPOCHS: %d, BATCH SIZE: %d, DEVICES: %s, SMOOTHING: %s" 
                % (args.lr, args.optim, args.num_epochs, args.batch_size, args.device, str(args.smoothing)))
    
    with open(save_path.argument_path, "w") as fw:
        json.dump(args.__dict__, fw, indent=2)


    batch_size = args.batch_size
    lr = args.lr
    smoothing = args.smoothing
    
    
    # gbm_tea/run.py (修改后的正确代码)

    traindata = GBMGraphDataset(data_dir="./GBM/postdata/train_test_split/train", 
                              sample_info_csv="./GBM/clinical_data_processed.csv",
                              is_train=True) 
    
    testdata = GBMGraphDataset(data_dir="./GBM/postdata/train_test_split/test", 
                              sample_info_csv="./GBM/clinical_data_processed.csv", # <--- 必须改回这个
                              is_train=False)
    train_iter = DataLoader(traindata, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(testdata, batch_size=batch_size, shuffle=False)
    
    CE_loss = nn.CrossEntropyLoss(reduction="mean",label_smoothing=smoothing)

    if args.optim == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    elif args.optim == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=0.01)
    elif args.optim == "ADAMW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    elif args.optim == "SGDNesterov":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, nesterov=True, weight_decay=0.01, momentum=0.01)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.01)
    
    # set scheduler
    if args.scheduler == "MULTILR":
        scheduler = MultiStepLR(optimizer, milestones=[int(s) for s in args.milestones.split("_")], gamma=args.lrgamma)
    elif args.scheduler == "CosineAnnealingWarmRestarts":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult)
    elif args.scheduler == "OneCycleLR":
        scheduler = OneCycleLR(optimizer, max_lr=args.max_lr, epochs=args.num_epochs, steps_per_epoch=len(train_iter))
    elif args.scheduler == "CyclicLR":
        scheduler = CyclicLR(optimizer, base_lr=lr, max_lr=args.max_lr, mode="exp_range", step_size_up=50, gamma=args.lrgamma)
    elif args.scheduler == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max)
    elif args.scheduler == "ExponentialLR":
        scheduler = ExponentialLR(optimizer, args.lrgamma)
    elif args.scheduler == "SequentialLR_1":
        sch1 = CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult)
        sch2 = ConstantLR(optimizer, factor=0.1, total_iters=args.num_epochs)
        scheduler = SequentialLR(optimizer, [sch1, sch2], milestones=[int(args.milestones.split("_")[0])])
    elif args.scheduler == "SequentialLR_2":    
        sch1 = OneCycleLR(optimizer, max_lr=args.max_lr, epochs=args.num_epochs, steps_per_epoch=len(train_iter))
        sch2 = ConstantLR(optimizer, factor=0.1, total_iters=args.num_epochs)
        scheduler = SequentialLR(optimizer, [sch1, sch2], milestones=[int(args.milestones.split("_")[0])])
    else:
        scheduler = None 
    


    # record model architecture
    logger = logger_initial(logger, model_architecture)
    train_args = TrainingConfig(logger, writer, save_path, args)
    
    train(model, train_iter, test_iter, CE_loss, optimizer, scheduler, train_args)




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, dest="lr", default=0.001, help="LEARNING RATE")
    parser.add_argument("--num_epochs", type=int, dest="num_epochs", default=200, help="NUM EPOCHS")
    parser.add_argument("--device", type=str, dest="device", default="cuda", help="DEVICE")
    parser.add_argument("--seed", type=int, dest="seed", default=None, help="RANDOM SEED")
    parser.add_argument("--batch_size", type=int, dest="batch_size", default=32, help="BATCH_SIZE")
    parser.add_argument("--optim", type=str, dest="optim", default="adam", help="OPTIMIZER")

    
    parser.add_argument("--savedisk", type=bool, dest="savedisk", default=False, help="SAVE INTERMEDIATE OUTPUT")
    
    parser.add_argument("--milestone", type=str, dest="milestones", default="30_50", help="MILESTONES for MULTIPLE LR")
    parser.add_argument("--lrgamma", type=float, dest="lrgamma", default=0.001, help="DECAY RATE OF LR")
    
    parser.add_argument("--T_0", type=int, dest="T_0", default=10, help="T_0 FOR CosineAnnealingWarmRestarts")
    parser.add_argument("--T_mult", type=int, dest="T_mult", default=2, help="T_mult FOR CosineAnnealingWarmRestarts")

    parser.add_argument("--T_max", type=int, dest="T_max", default=5, help="T_max FOR CosineAnnealing")
    parser.add_argument("--max_lr", type=float, dest="max_lr", default=0.001, help="MAX LR FOR CyclicLR & OneCycleLR")
    parser.add_argument("--scheduler", type=str, dest="scheduler", default="CosineAnnealingLR", help="ACTIVATE scheduler")
    parser.add_argument("--last_epoch", type=int, dest="last_epoch", default=10, help="LAST EPOCH FOR EVAL")

    parser.add_argument("--smoothing", type=float, dest="smoothing", default=0.03, help="ALPHA FOR LABEL SMOOTHING")

    args = parser.parse_args()

    return args
    

if __name__ == "__main__":

    args = parse_args()
    if args.seed is not None:
        setup_seed(args.seed)
        
    
    args.device = "cuda" if torch.cuda.is_available() else "cpu"


    time_tab = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("Running time tab:%s" % time_tab)
    

    # activate tensorboard
    log_dir = f"./TrainProcess/{time_tab}"
    command_list = ["tensorboard", "--logdir", log_dir]
    process = subprocess.Popen(command_list)

    # set model architecture
    model_architecture = params_setup()
    # run.py (修改后的正确代码)
    model = TwinsGCN(coord_in_channels=2,  # 明确指定坐标分支的输入维度为 2
                 feature_in_channels=model_architecture["in_chans"][0], 
                 heads=model_architecture["heads"][0],
                 out_channels=model_architecture["out_chans"][0],
                 class_num=4)

    main(args, model, model_architecture, time_tab)
 
    time.sleep(30)
    process.kill()