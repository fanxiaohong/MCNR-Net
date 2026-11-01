import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
import torch.nn as nn
import numpy as np
from argparse import ArgumentParser
import torchvision
from data_loader_HU import *
from model_fan_MSPD_share_mvmsunet2_scale4 import PrimalNet,DualNet,DualNet_h
from solver_CT_together_fan_iterator_MSPD import Solver_CT
###########################################################################################
# parameter
parser = ArgumentParser(description='CT-new-together')
parser.add_argument('--net_name', type=str, default='CT-fan-MSPD-share-fan-together-limited', help='name of net')
parser.add_argument('--model_dir', type=str, default='model and result fan CT HU', help='model_MRI,model_CT,trained or pre-trained model directory')
parser.add_argument('--log_dir', type=str, default='log_CT', help='log_MRI,log_CT')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=250, help='epoch number of end training')
parser.add_argument('--batch_size', type=int, default=1, help='CT=1')
parser.add_argument('--layer_num', type=int, default=7, help='D,7')
parser.add_argument('--num_features', type=int, default=32, help='G,32')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--ds_factor', type=int, default=6, help='{64,32,16, 8, 4} for sparse, [12, 6, 4, 3, 2.4] for limited')
parser.add_argument('--projection_mode', type=str, default='limited', help='sparse, limited')
parser.add_argument('--num_work', type=int, default=4, help='4,1')
parser.add_argument('--print_flag', type=int, default=1, help='print parameter number 1 or 0')
parser.add_argument('--result_dir', type=str, default='result', help='save result directory')
parser.add_argument('--CT_test_name', type=str, default='CT_test-HU-fan', help='save name of CT test set')
parser.add_argument('--together_mode', type=str, default='all', help='all,single')
parser.add_argument('--run_mode', type=str, default='test', help='train、test')
parser.add_argument('--save_image_mode', type=int, default=0, help='save 1, not 0 ,2 npy in test')
parser.add_argument('--loss_mode', type=str, default='L1-ssim', help='midloss-20k,midloss,L1,L2,Ista-net,Fista-net')
parser.add_argument('--penalty_mode', type=str, default='None', help='Fista-net,Weight,None')
args = parser.parse_args()
#########################################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
###########################################################################################
# data loading
train_loader,val_loader,test_loader = CT_dataloader(args.num_work,args.batch_size,args.run_mode)
model_i = PrimalNet(args.num_features)
model_r = DualNet(args.num_features)
model_rf = DualNet_h(args.num_features)
###################################################################################
# model
model_i = nn.DataParallel(model_i)
model_r = nn.DataParallel(model_r)
model_rf = nn.DataParallel(model_rf)
model_i = model_i.to(device)
model_r = model_r.to(device)
model_rf = model_rf.to(device)
###################################################################################
if args.print_flag:  # print networks parameter number
    print("Total number of paramerters in networks i is {}  ".format(sum(x.numel() for x in model_i.parameters())))
    print("Total number of paramerters in networks r is {}  ".format(sum(x.numel() for x in model_r.parameters())))
    print("Total number of paramerters in networks rf is {}  ".format(sum(x.numel() for x in model_rf.parameters())))

    # 联合打印两个模型的参数总和
    total_params = sum(p.numel() for p in model_i.parameters()) + sum(p.numel() for p in model_r.parameters()) + sum(p.numel() for p in model_rf.parameters())
    print("Total number of parameters in model_i + model_r + model_rf: {} ".format(total_params))
#######################################################################################
solver = Solver_CT(model_i,model_r, model_rf, train_loader,val_loader,test_loader, args, device, args.penalty_mode)
if args.run_mode == 'train':
    solver.train()
elif args.run_mode == 'test':
    solver.test()
#########################################################################################
