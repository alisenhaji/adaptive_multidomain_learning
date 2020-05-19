import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import models
import os
import time
import argparse
import numpy as np

from torch.autograd import Variable

import imdbfolder_coco as imdbfolder
import config_task
import utils_pytorch
import sgd

parser = argparse.ArgumentParser(description='PyTorch Residual Adapters training')
parser.add_argument('--dataset', default='cifar100', nargs='+', help='Task(s) to be trained')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--wd', default=1., type=float, help='weight decay for the classification layer')
parser.add_argument('--wd3x3', default=1., type=float, nargs='+', help='weight decay for the 3x3')
parser.add_argument('--wd1x1', default=1., type=float, nargs='+', help='weight decay for the 1x1')
parser.add_argument('--nb_epochs', default=120, type=int, help='nb epochs')
parser.add_argument('--step1', default=80, type=int, help='nb epochs before first lr decrease')
parser.add_argument('--step2', default=100, type=int, help='nb epochs before second lr decrease')
parser.add_argument('--mode', default='parallel_adapters', type=str, help='Task adaptation mode')
parser.add_argument('--proj', default='11', type=str, help='Position of the adaptation module')
parser.add_argument('--dropout', default='00', type=str, help='Position of dropouts')
parser.add_argument('--expdir', default='/scratch/shared/nfs1/srebuffi/exp/dem_learning/tmp/', help='Save folder')
parser.add_argument('--datadir', default='/scratch/local/ramdisk/srebuffi/decathlon/', help='folder containing data folder')
parser.add_argument('--imdbdir', default='/scratch/local/ramdisk/srebuffi/decathlon/annotations/', help='annotation folder')
parser.add_argument('--source', default='/scratch/shared/nfs1/srebuffi/exp/dem_learning/C100_alone/checkpoint/ckptpost11bnresidual11cifar1000.000180607060.t7', type=str, help='Network source')
parser.add_argument('--seed', default=0, type=int, help='seed')
parser.add_argument('--factor', default='1.', type=float, help='Width factor of the network')
args = parser.parse_args()
args.archi ='default'
config_task.mode = args.mode
config_task.proj = args.proj
config_task.factor = args.factor
args.use_cuda = torch.cuda.is_available()
if type(args.dataset) is str:
    args.dataset = [args.dataset]

if type(args.wd3x3) is float:
    args.wd3x3 = [args.wd3x3]

if type(args.wd1x1) is float:
    args.wd1x1 = [args.wd1x1]

if not os.path.isdir(args.expdir):
    os.mkdir(args.expdir) 

config_task.decay3x3 = np.array(args.wd3x3) * 0.0001
config_task.decay1x1 = np.array(args.wd1x1) * 0.0001
args.wd = args.wd *  0.0001

args.ckpdir = args.expdir + '/checkpoint/'
args.svdir  = args.expdir + '/results/'

if not os.path.isdir(args.ckpdir):
    os.mkdir(args.ckpdir) 

if not os.path.isdir(args.svdir):
    os.mkdir(args.svdir) 

config_task.isdropout1 = (args.dropout[0] == '1')
config_task.isdropout2 = (args.dropout[1] == '1')

#####################################

# Prepare data loaders
train_loaders, val_loaders, num_classes, min_lab = imdbfolder.prepare_data_loaders(args.dataset,args.datadir,args.imdbdir,True)
args.num_classes = num_classes


torch.nn.Module.dump_patches = True


# Load checkpoint and initialize the networks with the weights of a pretrained network
print('==> Resuming from checkpoint..')
checkpoint = torch.load(args.source, encoding='latin1')
net_old = checkpoint['net']
net = models.resnet26(num_classes)
store_data = []
for name, m in net_old.named_modules():
    if isinstance(m, nn.Conv2d) and (m.kernel_size[0]==3):
        store_data.append(m.weight.data)

element = 0
for name, m in net.named_modules():
    if isinstance(m, nn.Conv2d) and (m.kernel_size[0]==3):
        m.weight.data = store_data[element]
        element += 1

store_data = []
store_data_bias = []
store_data_rm = []
store_data_rv = []
names = []

for name, m in net_old.named_modules():
    if isinstance(m, nn.BatchNorm2d) and 'bns.' in name:
        names.append(name)
        store_data.append(m.weight.data)
        store_data_bias.append(m.bias.data)
        store_data_rm.append(m.running_mean)
        store_data_rv.append(m.running_var)

# Special case to copy the weight for the BN layers when the target and source networks have not the same number of BNs
import re
condition_bn = 'noproblem'
if len(names) != 51 and args.mode == 'series_adapters':
    condition_bn ='bns.....conv'

for id_task in range(len(num_classes)):
    element = 0
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d) and 'bns.'+str(id_task) in name and not re.search(condition_bn,name):
                m.weight.data = store_data[element].clone()
                m.bias.data = store_data_bias[element].clone()
                m.running_var = store_data_rv[element].clone()
                m.running_mean = store_data_rm[element].clone()
                element += 1

#net.linears[0].weight.data = net_old.linears[0].weight.data
#net.linears[0].bias.data = net_old.linears[0].bias.data

del net_old

start_epoch = 0
best_acc = [0,0,0]  # best test accuracy

results1 = np.zeros((4,start_epoch+args.nb_epochs,len(args.num_classes)))
results2 = np.zeros((4,start_epoch+args.nb_epochs,len(args.num_classes)))
results3 = np.zeros((4,start_epoch+args.nb_epochs,len(args.num_classes)))



all_tasks = range(len(args.dataset))
np.random.seed(1993)

if args.use_cuda:
    net.cuda()
    cudnn.benchmark = True


# Freeze 3*3 convolution layers
for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) and (m.kernel_size[0]==3):
            m.weight.requires_grad = False


exp_name = '_cnns'


print(net)

args.criterion = nn.CrossEntropyLoss()
optimizer = sgd.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.wd)

#print(net)
print("Start training")
for epoch in range(start_epoch, start_epoch+args.nb_epochs):
    training_tasks = utils_pytorch.adjust_learning_rate_and_learning_taks(optimizer, epoch, args)
    st_time = time.time()
    
    # Training and validation
    train_acc1, train_loss1, train_acc2, train_loss2, train_acc3, train_loss3 = utils_pytorch.train(epoch, train_loaders, training_tasks, net, args, optimizer)
    test_acc1, test_loss1, test_acc2, test_loss2, test_acc3, test_loss3  = utils_pytorch.test(epoch,val_loaders, all_tasks, net, best_acc, args, optimizer, exp_name)
        
    # Record statistics
    for i in range(len(training_tasks)):
        current_task = training_tasks[i]
        results1[0:2,epoch,current_task] = [train_loss1[i],train_acc1[i]]

    for i in all_tasks:
        results1[2:4,epoch,i] = [test_loss1[i],test_acc1[i]]
        
    for i in range(len(training_tasks)):
        current_task = training_tasks[i]
        results2[0:2,epoch,current_task] = [train_loss2[i],train_acc2[i]]

    for i in all_tasks:
        results2[2:4,epoch,i] = [test_loss2[i],test_acc2[i]]
    
    for i in range(len(training_tasks)):
        current_task = training_tasks[i]
        results3[0:2,epoch,current_task] = [train_loss3[i],train_acc3[i]]

    for i in all_tasks:
        results3[2:4,epoch,i] = [test_loss3[i],test_acc3[i]]
    
    np.save(args.svdir+'/results_'+''.join(args.dataset)+'_Exit1' + exp_name,results1)
    np.save(args.svdir+'/results_'+''.join(args.dataset)+'_Exit2'+ exp_name,results2)
    np.save(args.svdir+'/results_'+''.join(args.dataset)+'_Exit3'+ exp_name,results3)
    print('Epoch lasted {0}'.format(time.time()-st_time))

######################################
    
from collections import OrderedDict
import json


#model =  models.resnet26(num_classes)
#model.load_state_dict(torch.load(args.ckpdir+'/'+''.join(args.dataset)+'_'+exp_name+'.pth'))
#if args.use_cuda:
#    model.cuda()
#    cudnn.benchmark = True


test_loaders, timgnames_id = imdbfolder.prepare_data_loaders_online(args.dataset,args.datadir,args.imdbdir,False)
predictions1, predictions2, predictions3 = utils_pytorch.test_online(test_loaders, all_tasks, net, best_acc, args, optimizer, min_lab)

#print(predictions1[0], timgnames_id[0])
res_exit1 = []
for i in range(len(predictions1)):
    #res.append('{"image_id":'+ str(timgnames_id[i]) +',"category_id":' + str(predictions[i]) +'}')
    #res.append({"image_id": timgnames_id[i], "category_id": predictions[i]})
    res_exit1.append(OrderedDict([('image_id', timgnames_id[i]), ('category_id', predictions1[i])]))
    
res_exit2 = []
for i in range(len(predictions2)):
    res_exit2.append(OrderedDict([('image_id', timgnames_id[i]), ('category_id', predictions2[i])]))
    
res_exit3 = []
for i in range(len(predictions3)):
    res_exit3.append(OrderedDict([('image_id', timgnames_id[i]), ('category_id', predictions3[i])]))
    

# Appends to the json file
with open('save/results_'+''.join(args.dataset)+'_'+ exp_name +'_exit1.json', 'w') as outfile:
    json.dump(res_exit1, outfile)

with open('save/results_'+''.join(args.dataset)+'_'+ exp_name +'_exit2.json', 'w') as outfile:
    json.dump(res_exit2, outfile)

with open('save/results_'+''.join(args.dataset)+'_'+ exp_name +'_exit3.json', 'w') as outfile:
    json.dump(res_exit3, outfile)