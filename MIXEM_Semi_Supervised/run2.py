

import sys,os,time,argparse
import yaml
from train import MIXEM
import torch
#from linear import *

import socket

#Naai imports
import sys, os,argparse, pickle,time
import torch
torch.manual_seed(179510)
import numpy as np
np.random.seed(179510)
import yaml
from sklearn import preprocessing
import importlib.util

from tqdm import tqdm
import torch.nn as nn

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

import os
from PIL import Image
import matplotlib.pyplot as plt
import shutil

### CUSTOM DATASET
class NewDataset(torch.utils.data.Dataset):

    def __init__(self, path, mode= 1): #1 - supervised

        self.mode = mode
        self.path = path
        imgs = os.listdir(path)

        X = []

        if mode == 1:
            Y = []

            for t in imgs:
                X.append(t)
                Y.append(int(t[:-5][-1]))
            

            self.X = X
            self.Y = Y
            self.length = len(X)

        else:

            for t in imgs:
                X.append(t)
            
            self.X = X
            self.length = len(X)
            
    
    def __len__(self):
        return self.length 

    def __getitem__(self, ind):

        img_path = self.path + self.X[ind]
        img = Image.open(img_path)

        
        img = transforms.ToTensor()(img)
        
        if self.mode == 1:
            img_label = self.Y[ind]
            return img, img_label

        else: #unsupervised
            img_aug = transforms.RandomRotation(45)(img)
            return img, img_aug

def our_dataloader(train_path, val_path, batch_size= 2, mode= 0): #0 - unsupersized, 1 - supervised

    train_data = NewDataset(train_path, mode= mode)
    val_data  = NewDataset(val_path, mode= mode)

    train_loader = DataLoader(train_data,
                              batch_size= batch_size, 
                              num_workers= 4,
                              shuffle= True, 
                              drop_last= True)
    val_loader = DataLoader(val_data,
                            batch_size= batch_size, 
                            num_workers= 4,
                            shuffle= False, 
                            drop_last= True)
    return train_loader,len(train_data), val_loader, len(val_data)  

def our_dataloader_new(train_path_clean, val_path_clean, train_path_unclean, val_path_unclean, batch_size= 2): #0 - unsupersized, 1 - supervised

    train_data_clean = NewDataset(train_path_clean, mode= 1)
    val_data_clean  = NewDataset(val_path_clean, mode= 1)

    train_data_unclean = NewDataset(train_path_unclean, mode= 0)
    val_data_unclean  = NewDataset(val_path_unclean, mode= 0)

    train_loader_clean = DataLoader(train_data_clean,
                                    batch_size= batch_size, 
                                    num_workers= 4,
                                    shuffle= True,
                                    drop_last= True)

    val_loader_clean = DataLoader(val_data_clean,
                                  batch_size= batch_size, 
                                  num_workers= 4,
                                  shuffle= False, 
                                  drop_last= True)

    train_loader_unclean = DataLoader(train_data_unclean,
                                      batch_size= batch_size, 
                                      num_workers= 4,
                                      shuffle= True, 
                                      drop_last= True)

    val_loader_unclean = DataLoader(val_data_unclean,
                                    batch_size= batch_size, 
                                    num_workers= 4,
                                    shuffle= False, 
                                    drop_last= True)
    return train_loader_clean, val_loader_clean, train_loader_unclean, val_loader_unclean
### END CUSTOM DATASET

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path',type=str, default=None)
    parser.add_argument('--log_dir',type=str, default=None)
    parser.add_argument('--gpus',type=str, default=None)
    parser.add_argument('--start_epoch',type=int, default=1)
    parser.add_argument('--master_batch_size',type=int, default=-1)
    parser.add_argument('--init_from',type=str, default=None)
    parser.add_argument('--dataroot',type=str, default=None)
    parser.add_argument('--lineardataroot',type=str, default=None)
    parser.add_argument('--dsname',type=str, default=None)
    parser.add_argument('--epochs',type=int, default=None)
    parser.add_argument('--num_workers',type=int, default=4)
    parser.add_argument('--fixed_lr',action='store_true')
    parser.add_argument('--no_eval',action='store_true')
    parser.add_argument('--resume',action='store_true')
    
    args = parser.parse_args(sys.argv[1:])
    
    for k,v in sorted(vars(args).items()):
        print('{}: {}'.format(str(k).ljust(20),v ) )
    ###################################################################################################

    if args.resume:
        args.config_path = os.path.join(args.log_dir,'checkpoints','config.yaml')
        with open(os.path.join(args.log_dir, 'checkpoints','checkpoint'),'r') as _if:
            args.init_from = _if.readline().strip('\n')
        print('init_from:',args.init_from)
            

    config = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)

    args.gpus = list(map(int, args.gpus.split(',')))

    _lr = config['learning_rate']
    config.update({'log_dir': args.log_dir, 
                    'config_path':args.config_path,
                    'gpus':args.gpus,   
                    'master_batch_size':args.master_batch_size,
                    'dataroot':args.dataroot,
                    'resume':args.resume,
                    'start_epoch':  int(args.init_from.split('.')[0].split('_')[-1])+1 if args.resume else args.start_epoch
                    })
    config.update({'lineardataroot': args.lineardataroot or config.get('lineardataroot',None)})
    config.update({'init_from': args.init_from or config['init_from']})
    config.update({'epochs':args.epochs or config['epochs']})
    config.update({'no_eval':args.no_eval or config.get('no_eval',False)})
    config.update({'fixed_lr':args.fixed_lr or config.get('fixed_lr',False)})

    config['dataset'].update({'dataset': args.dsname or config['dataset']['dataset']})
    dsname = config['dataset']['dataset']
    print('dsname:',config['dataset']['dataset'])
    config['dataset'][dsname].update({'num_workers': args.num_workers or config['dataset'][dsname]['num_workers']})

    config.update({'conv1_k3':config['conv1_k3'] or (dsname.startswith('CIFAR') )})
    if args.resume:
        config.update({'init_from': os.path.join(config['log_dir'],'checkpoints',config['init_from']) })
        assert os.path.exists(config['init_from'])
    
    if config['init_from'] == 'None':
        config.update({'init_from':None})
    
    model_config = '_{}_{}'.format(config['model']['base_model'],config['model']['out_dim'])
    con1k3 = '_conv1k3' if config['conv1_k3'] else ''
    model_config += con1k3
    
    temp = '_temp_{}'.format(config['loss']['temperature'])
    
    optconfig = '_LR_{}'.format(config['learning_rate'])
    if config['fixed_lr']:
        optconfig += '_FIXED'
    if config['weight_decay'] != 1e-5:
        optconfig += '_WDECAY_{}'.format(config['weight_decay']) 
    
    finetune = ''
    if not args.resume:
        finetune = '_init_form_{}_{}'.format(config['init_from'].split('/')[-3].split('_')[-1],config['init_from'].split('/')[-1] ) if config['init_from'] is not None else ''
        if args.start_epoch > 1:
            finetune += '_STARTEP_{}'.format(args.start_epoch)
    mixture = ''
    if 'Mixture' in config['model']:
        config['model']['Mixture'].update({'temperature': config['model']['Mixture'].get('temperature',1.),\
                                        'pimax_loss_w':config['model']['Mixture'].get('pimax_loss_w',0),\
                                        'w_push':config['model']['Mixture'].get('w_push',0),\
                                        'w_pull':config['model']['Mixture'].get('w_pull',0)
                                        })
        pimax_loss = ''
        if config['model']['Mixture']['pimax_loss_w']>0:
            pimax_loss = '_PIMAXL_{}'.format(config['model']['Mixture']['pimax_loss_w'])
        pushw = config['model']['Mixture']['w_push']
        pullw = config['model']['Mixture']['w_pull']
        pp = 'PP_{}_{}'.format(pushw,pullw) if pushw > 0 or pullw > 0 else ''
            
            

        mixture = '_MIXTURE_{}_{}_ENT_{}{}'.format(config['model']['Mixture']['n_comps'],\
                                            pp,config['model']['Mixture']['w_entropy'],pimax_loss)
                    


    log_dir_label = '{}{}{}_{}{}{}{}'.format(dsname,mixture,\
                                                model_config,\
                                                config['batch_size'],\
                                                temp,optconfig,finetune)
    if args.resume:
        log_dir = config['log_dir']
        if log_dir.endswith('/'):
            log_dir = log_dir[:-1]
        print('log_dir:'.ljust(30),log_dir.split('/')[-1])
        print('log_dir_label:'.ljust(30),log_dir_label)
        assert log_dir.split('/')[-1].startswith(log_dir_label)
    else:
        log_dir = '{}/{}_gpus_{}_{}'.format(config['log_dir'],log_dir_label,','.join(map(str,args.gpus)),int(time.time()))
        config.update({'log_dir':log_dir})


    threshold_name = 95
    folderpath = '/content/drive/MyDrive/Fall 2021/18786 Introduction to Deep Learning/Project/Naayes/T' + str(threshold_name)
    try:
        os.mkdir(folderpath)
    except:
        print("Folder exists")
        shutil.rmtree(folderpath)
        os.mkdir(folderpath)
    print("Folder created")

    for i in range(0, 26):

        print("ITERATION: ", i, "-"*80)
        print("* "*80)
        print(" *"*80)
        clean_dir = folderpath + '/clean_dir_' + str(i)
        unclean_dir = folderpath + '/unclean_dir_' + str(i)
        #clean_dir = '/content/drive/MyDrive/Fall 2021/18786 Introduction to Deep Learning/Project/Naaye9/clean_naai_' + str(i)
        #unclean_dir = '/content/drive/MyDrive/Fall 2021/18786 Introduction to Deep Learning/Project/Naaye9/dirty_naai_' + str(i)
        
        os.mkdir(clean_dir)
        os.mkdir(unclean_dir)

        clean_dir += '/'
        unclean_dir += '/'

        
        if i == 0:
            init_path = '/content/drive/MyDrive/Fall 2021/18786 Introduction to Deep Learning/Project/MIXEM/STL10_800.pth'
        else:
            init_path = '/content/drive/MyDrive/Fall 2021/18786 Introduction to Deep Learning/Project/Naaye9/Model_' + str(i) + '.pth'
        
        config['init_from'] = init_path
        print("\nModel Path: ", init_path)

        config['epochs'] = 2
        mixem = MIXEM(config)
        mixem._gen_data(clean_dir, unclean_dir, threshold= threshold_name/100)

        tc, vc, tu, vu = our_dataloader_new(clean_dir, clean_dir, unclean_dir, unclean_dir, batch_size= 512)
        print("Got loaders da echa:")
        

        mixem.train_all(tc, vc, tu, vu, 512, model_no = i+1)



    # for i in range(26, 101, 2):
    #     print("\nMODEL " + str(i) + "-"*80)
    #     init_path = '/content/drive/MyDrive/Fall 2021/18786 Introduction to Deep Learning/Project/MIXEM/log/Models 2/checkpoints/STL10_' + str(i) + '.pth'
    #     print("\nModel Path: ", init_path)
    #     config['init_from'] = init_path
    #     mixem = MIXEM(config)
    #     mixem._gen_data(threshold= 0.80)
    

    # config['epochs'] = 50 #25 passes
    # print('log_dir:',config['log_dir'])
    # mode = 1




    # #give the Mnist paths
    # train_path_clean = '/content/drive/MyDrive/Fall 2021/18786 Introduction to Deep Learning/Project/Naaye9/clean_naai/'
    # val_path_clean = '/content/drive/MyDrive/Fall 2021/18786 Introduction to Deep Learning/Project/Naaye9/clean_naai_val/'
    # train_path_unclean = '/content/drive/MyDrive/Fall 2021/18786 Introduction to Deep Learning/Project/Naaye9/dirty_naai/'
    # val_path_unclean = '/content/drive/MyDrive/Fall 2021/18786 Introduction to Deep Learning/Project/Naaye9/dirty_naai_val/'
    
    # batch_size = 512

    # #init_path = '/content/drive/MyDrive/Fall 2021/18786 Introduction to Deep Learning/Project/MIXEM/log/Models 2/checkpoints/STL10_' + str(i) + '.pth'
    # init_path = '/content/drive/MyDrive/Fall 2021/18786 Introduction to Deep Learning/Project/MIXEM/STL10_800.pth'
    # print("\nModel Path: ", init_path)
    # config['init_from'] = init_path
    # #tl, lt, vl, lv = our_dataloader(train_path_unclean, train_path_unclean, batch_size= batch_size, mode= 0)
    # #tc, vc, tu, vu = our_dataloader_new(train_path_clean, val_path_clean, train_path_unclean, val_path_unclean, batch_size= batch_size)
    # #print("Got loaders da")

    # mixem = MIXEM(config)
    # #mixem.train(tl, vl, batch_size, mode= 0)
    # #mixem.train_all(tc, vc, tu, vu, batch_size)
    # mixem._gen_data(train_path_clean, train_path_unclean, threshold= 0.90)

if __name__ == "__main__":

    main()
    print('End')
