import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
import time
import numpy as np
from torch.autograd import Variable
import config_task

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate_and_learning_taks(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every X epochs"""
    if epoch >= args.step2: 
        lr = args.lr * 0.01
    elif epoch >= args.step1:
        lr = args.lr * 0.1
    else:
        lr = args.lr
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Return training classes
    return range(len(args.dataset))

# Training
def train(epoch, tloaders, tasks, net, args, optimizer,list_criterion=None):
    print('\nEpoch: %d' % epoch)
    net.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses1 = [AverageMeter() for i in tasks]
    top11 = [AverageMeter() for i in tasks]
    
    losses2 = [AverageMeter() for i in tasks]
    top12 = [AverageMeter() for i in tasks]
    
    losses3 = [AverageMeter() for i in tasks]
    top13 = [AverageMeter() for i in tasks]
    
    end = time.time()
    
    loaders = [tloaders[i] for i in tasks]
    min_len_loader = np.min([len(i) for i in loaders])
    train_iter = [iter(i) for i in loaders]
        
    for batch_idx in range(min_len_loader*len(tasks)):
        config_task.first_batch = (batch_idx == 0)
        # Round robin process of the tasks
        current_task_index = batch_idx % len(tasks)
        inputs, targets = (train_iter[current_task_index]).next()
        config_task.task = tasks[current_task_index]
        # measure data loading time
        data_time.update(time.time() - end)
        if args.use_cuda:
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        optimizer.zero_grad()
        
        inputs, targets = Variable(inputs), Variable(targets)
        outputs1 , outputs2, outputs3 = net(inputs)
        
        loss1 = args.criterion(outputs1, targets)
        loss2 = args.criterion(outputs2, targets)
        loss3 = args.criterion(outputs3, targets)
        
        # measure accuracy and record loss
        (losses1[current_task_index]).update(loss1.data.item(), targets.size(0))
        _, predicted1 = torch.max(outputs1.data, 1)
        correct1 = predicted1.eq(targets.data).cpu().sum()
        (top11[current_task_index]).update(correct1*100./targets.size(0), targets.size(0))     
        
        (losses2[current_task_index]).update(loss2.data.item(), targets.size(0))
        _, predicted2 = torch.max(outputs2.data, 1)
        correct2 = predicted2.eq(targets.data).cpu().sum()
        (top12[current_task_index]).update(correct2*100./targets.size(0), targets.size(0))       
        
        (losses3[current_task_index]).update(loss3.data.item(), targets.size(0))
        _, predicted3 = torch.max(outputs3.data, 1)
        correct3 = predicted3.eq(targets.data).cpu().sum()
        (top13[current_task_index]).update(correct3*100./targets.size(0), targets.size(0))        
        
        # apply gradients   
        loss1.backward(retain_graph=True)
        loss2.backward(retain_graph=True)
        loss3.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if batch_idx % 200 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                   epoch, batch_idx, min_len_loader*len(tasks), batch_time=batch_time,
                   data_time=data_time))
            for i in range(len(tasks)):
                print('[Exit 1] Task {0} : Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {top1.val:.3f} ({top1.avg:.3f})'.format(tasks[i], loss=losses1[i], top1=top11[i]))
                print('[Exit 2] Task {0} : Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {top1.val:.3f} ({top1.avg:.3f})'.format(tasks[i], loss=losses2[i], top1=top12[i]))
                print('[Exit 3] Task {0} : Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {top1.val:.3f} ({top1.avg:.3f})'.format(tasks[i], loss=losses3[i], top1=top13[i]))
        
    return [top11[i].avg for i in range(len(tasks))], [losses1[i].avg for i in range(len(tasks))], [top12[i].avg for i in range(len(tasks))], [losses2[i].avg for i in range(len(tasks))] , [top13[i].avg for i in range(len(tasks))], [losses3[i].avg for i in range(len(tasks))]

def test(epoch, loaders, all_tasks, net, best_acc, args, optimizer,exp_name):
    net.eval()
    losses1 = [AverageMeter() for i in all_tasks]
    top11 = [AverageMeter() for i in all_tasks]
    
    losses2 = [AverageMeter() for i in all_tasks]
    top12 = [AverageMeter() for i in all_tasks]
    
    losses3 = [AverageMeter() for i in all_tasks]
    top13 = [AverageMeter() for i in all_tasks]
    
    
    print('Epoch: [{0}]'.format(epoch))
    for itera in range(len(all_tasks)):
        i = all_tasks[itera]
        config_task.task = i
        for batch_idx, (inputs, targets) in enumerate(loaders[i]):
            if args.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                inputs, targets = Variable(inputs), Variable(targets)
                outputs1, outputs2, outputs3 = net(inputs)
                
                if isinstance(outputs1, tuple):
                     outputs1 = outputs1[0]
                loss1 = args.criterion(outputs1, targets)
                
                losses1[itera].update(loss1.data.item(), targets.size(0))
                _, predicted1 = torch.max(outputs1.data, 1)
                
                if isinstance(outputs2, tuple):
                     outputs2 = outputs2[0]
                loss2 = args.criterion(outputs2, targets)
                
                losses2[itera].update(loss2.data.item(), targets.size(0))
                _, predicted2 = torch.max(outputs2.data, 1)
                
                if isinstance(outputs3, tuple):
                     outputs3 = outputs3[0]
                loss3 = args.criterion(outputs3, targets)

                losses3[itera].update(loss3.data.item(), targets.size(0))
                _, predicted3 = torch.max(outputs3.data, 1)
                

            correct1 = predicted1.eq(targets.data).cpu().sum()
            top11[itera].update(correct1*100./targets.size(0), targets.size(0))
            
            correct2 = predicted2.eq(targets.data).cpu().sum()
            top12[itera].update(correct2*100./targets.size(0), targets.size(0))
            
            correct3 = predicted3.eq(targets.data).cpu().sum()
            top13[itera].update(correct3*100./targets.size(0), targets.size(0))
        
        print('[Exit 1] Task {0} : Test Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Test Acc {top1.val:.3f} ({top1.avg:.3f})'.format(i, loss=losses1[itera], top1=top11[itera]))
        
        print('[Exit 2] Task {0} : Test Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Test Acc {top1.val:.3f} ({top1.avg:.3f})'.format(i, loss=losses2[itera], top1=top12[itera]))
        
        print('[Exit 3] Task {0} : Test Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Test Acc {top1.val:.3f} ({top1.avg:.3f})'.format(i, loss=losses3[itera], top1=top13[itera]))
    
    # Save checkpoint.
    acc1 = np.sum([top11[i].avg for i in range(len(all_tasks))])
    if acc1 > best_acc[0]:
        print('Saving exit 1..')
        torch.save(net.state_dict(), args.ckpdir+'/'+''.join(args.dataset)+'_'+exp_name+'_exit1.pth')
        best_acc[0] = acc1
        
    acc2 = np.sum([top12[i].avg for i in range(len(all_tasks))])
    if acc2 > best_acc[1]:
        print('Saving exit 2..')
        torch.save(net.state_dict(), args.ckpdir+'/'+''.join(args.dataset)+'_'+exp_name+'_exit2.pth')
        best_acc[1] = acc2
        
    acc3 = np.sum([top13[i].avg for i in range(len(all_tasks))])
    if acc3 > best_acc[2]:
        print('Saving exit 3..')
        torch.save(net.state_dict(), args.ckpdir+'/'+''.join(args.dataset)+'_'+exp_name+'_exit3.pth')
        best_acc[2] = acc3       
    
    return [top11[i].avg for i in range(len(all_tasks))], [losses1[i].avg for i in range(len(all_tasks))], [top12[i].avg for i in range(len(all_tasks))], [losses2[i].avg for i in range(len(all_tasks))], [top13[i].avg for i in range(len(all_tasks))], [losses3[i].avg for i in range(len(all_tasks))]

def test_online(loaders, all_tasks, net, best_acc, args, optimizer, min_lab):
    net.eval()
    print('Evaluating online test set ...')
    predictions1 = []
    predictions2 = []
    predictions3 = []
    for itera in range(len(all_tasks)):
        i = all_tasks[itera]
        config_task.task = i
        for batch_idx, (inputs) in enumerate(loaders[i]):
            if args.use_cuda:
                inputs = inputs.cuda()
            with torch.no_grad():
                inputs = Variable(inputs)
                outputs1, outputs2, outputs3 = net(inputs)

                if isinstance(outputs1, tuple):
                     outputs1 = outputs1[0]
                
                _, predicted1 = torch.max(outputs1.data, 1)
                
                if isinstance(outputs2, tuple):
                     outputs2 = outputs2[0]
                
                _, predicted2 = torch.max(outputs2.data, 1)
                
                if isinstance(outputs3, tuple):
                     outputs3 = outputs3[0]

                _, predicted3 = torch.max(outputs3.data, 1)

                correct1 = predicted1 + min_lab + 1
                predictions1.extend(correct1.tolist())
                
                correct2 = predicted2 + min_lab + 1
                predictions2.extend(correct2.tolist())
                               
                correct3 = predicted3 + min_lab + 1
                predictions3.extend(correct3.tolist())
                
    return predictions1, predictions2, predictions3