import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
import sys

from Models.alexnet import AlexNet
from Models.resnet import *
from Models.models import *
from Server.ServerAvg import FedAvg
from Server.ServerAvgBackdoor import FedAvgBackdoor


import torch
torch.manual_seed(0)

def main(dataset, algorithm,batch_size, learning_rate, lamda, num_glob_iters,
         local_epochs, optimizer, numusers, K, times, gpu,n_attackers,attacker_type,backdoor,target_label,attack_epoch):

    # Get device status: Check GPU or CPU
    device = torch.device("cuda")

    for i in range(times):
        print("---------------Running time:------------",i)
        if dataset=="Cifar100":
            model = AlexNet(num_classes=20).to(device)
        # select algorithm
        if(algorithm == "FedAvg"):
            server = FedAvgBackdoor(device, dataset, algorithm, model, batch_size, learning_rate,num_glob_iters, local_epochs, optimizer, numusers, i,n_attackers,attacker_type,backdoor,target_label,attack_epoch)
        
        server.train()
        
        #server.test()

    # Average data 
    average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,learning_rate=learning_rate, beta = beta, algorithms=algorithm, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate,times = times)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cifar100", choices=["Mnist", "Synthetic", "Cifar10","Cifar100"])
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Local learning rate")
    parser.add_argument("--lamda", type=int, default=15, help="Regularization term")
    parser.add_argument("--num_global_iters", type=int, default=110)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="FedAvg",choices=["pFedMe", "PerAvg", "FedAvg"]) 
    parser.add_argument("--numusers", type=int, default=50, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=5, help="Computation steps")
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to run the experiments, -1 mean CPU, 0,1,2 for GPU")
    parser.add_argument("--attacker_type", type=str, default="backdoor", choices=["backdoor"])
    parser.add_argument("--n_attackers", type=int, default=1, help="The number of attackers")
    parser.add_argument("--backdoor", type=int, default=90, help="backdoor label")
    parser.add_argument("--target_label", type=int, default=0, help="target label")
    parser.add_argument("--attack_epoch", type=int, default=5, help="attack epoch")
    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Subset of users      : {}".format(args.numusers))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Attacker Type       : {}".format(args.attacker_type))
    print("Number of attackers       : {}".format(args.n_attackers))
    print("Semantic Backdoor       : {}".format(args.backdoor))
    print("Target label of Backdoor       : {}".format(args.target_label))
    print("Attack epoch       : {}".format(args.attack_epoch))
    print("=" * 80)

    main(
        dataset=args.dataset,
        algorithm = args.algorithm,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate, 
        lamda = args.lamda,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer= args.optimizer,
        numusers = args.numusers,
        K=args.K,
        times = args.times,
        gpu=args.gpu,
        n_attackers=args.n_attackers,
        attacker_type=args.attacker_type,
        backdoor=args.backdoor,
        target_label=args.target_label,
        attack_epoch=args.attack_epoch
        )