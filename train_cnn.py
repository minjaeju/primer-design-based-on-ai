from __future__ import unicode_literals, print_function, division
import random

import time
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import KFold

from src.model_cnn import *
from src.data import *
from src.plot_utils import show_plot

import argparse
import datetime
from os import makedirs


parser = argparse.ArgumentParser(description='Parser for training.')
parser.add_argument('-k', '--k_folds', type=int, default=5,
                    help='number of folds for cross validation (default: 5)')
parser.add_argument('-e', '--num_epochs', type=int, default=3000,
                    help='number of epochs (default: 500)')
parser.add_argument('-b', '--batch_size', type=int, default=10,
                    help='size of batch (default: 10)')
parser.add_argument('-l', '--learning_rate', type=float, default=1e-5,
                    help='learning rate (default: 1e-5)')
parser.add_argument('-t', '--teacher_forcing_rate', type=float, default=0.5,
                    help='teacher forcing rate (default: 0.5)')
parser.add_argument('-d', '--dropout', type=float, default=0.3,
                    help='encoder dropout rate (default: 0.5)')
# parser.add_argument('--embedding_dim', type=int, default=24,
#                     help='embedding dimension (default: 24)')
parser.add_argument('--hidden_dim', type=int, default=16,
                    help='hidden dimension (default: 64)')
parser.add_argument('--plot_every', type=int, default=1,
                    help='number of epochs for plotting (default: 1)')
parser.add_argument('--print_every', type=int, default=100,
                    help='number of epochs for printing losses for plot (default: 100)')
parser.add_argument('--data_path', default='./data/train_df.csv',
                    help='path for train dataframe (default: ./data/train_df.csv')
parser.add_argument('--key', default='R primer',
                    help='sequence type for prediction (default: R primer)')
parser.add_argument('--word_dict', default='./data/word_dict.pkl',
                    help='path for word dict (default: ./data/word_dict.pkl)')
parser.add_argument('--kernel_size', type=int, default=3,
                    help='conv kernel size (default: 3)')
# parser.add_argument('--debug', type=bool, default=False,
#                     help='debug mode (default: False)')
parser.add_argument('--gpu', type=int, default=0,
                    help='gpu assigned (default: 0)')

args = parser.parse_args()
cur_date = datetime.datetime.now().strftime('%y%m%d-%H%M%S')


if __name__ == '__main__':
#     gpu = 1
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    k_folds = args.k_folds
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
#     plot_every = args.plot_every
#     emb_dim = args.embedding_dim
    hidn_dim = args.hidden_dim
    key = args.key
    max_seq_len = 40
    
    result_img_name = 'train'
    save_model_path = f'./model/cnn/{cur_date}/'
    save_result_path = f'./result/cnn/{cur_date}/'
    
    # Debug mode
#     if args.debug:
#         result_img_name += '_debug'
    
#     loss_function = nn.CrossEntropyLoss(ignore_index=PAD_token)
    loss_function = nn.MSELoss()
    results = {}
    loss_list = {}
    
    torch.manual_seed(42)
    
    dataset_train = pd.read_csv(args.data_path)  #.to_numpy()
#     with open(args.word_dict,'rb') as f:
#         word2index_dict = pickle.load(f)
#     word2index_dict = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'R':4, 'Y':5, 'M':6, 'K':7}
    word2index_dict = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    vocab_size = len(word2index_dict)
    
    dataset = Dataset_FRP(dataset_train, key)
#     dataset.set_max_seq_len()
    kfold = KFold(n_splits=k_folds, shuffle=True)
    
    # Make directory for saved model
    try:
        makedirs(save_model_path, exist_ok=True)
        makedirs(save_result_path, exist_ok=True)
    except:
        print("Error while making directories")
        raise

    print('Saved Model Path: %s' % save_model_path)

    print('----------------------------------')
    
    for fold, (train_index, test_index) in enumerate(kfold.split(dataset)):
        
        print(f'Fold {fold}')
        print('----------------------------------')
        
        # parameters for printing
        loss_plot_list = []
        loss_plot = 0.0
    
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_index)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_index)
    
        trainloader = get_loader_CNN(dataset, train_subsampler, batch_size, key, word2index_dict)
        testloader = get_loader_CNN(dataset, test_subsampler, batch_size, key, word2index_dict)
        
        model = CNN(max_seq_len, vocab_size, hidn_dim, device, args.kernel_size, args.dropout).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
#         loss_epoch = []

        for epoch in range(1, num_epochs+1):
            
#             print(f'Epoch {epoch+1}')
            loss = 0.0
                  
            for i, data in enumerate(trainloader, 0):
                inputs, targets = data
                outputs = model(inputs.to(device))
                outputs = outputs.view(-1).to(device)
                targets = targets.type(torch.FloatTensor).view(-1).to(device)
                
#                 print(outputs.size())
#                 print(outputs)
                losses = loss_function(outputs, targets)
                  
                losses.backward()
                optimizer.step()
                  
                loss += losses.item()
                ### Print 추가
                loss_plot += losses.item()
                
            ### Print 추가
            loss_plot_avg = loss_plot / (args.plot_every*(i+1))
            if epoch % args.print_every == 0:
                print('Epoch %d / %d (%d%%) Loss: %.4f' % (epoch, num_epochs, epoch / num_epochs * 100, loss_plot_avg))
                
            if epoch % args.plot_every == 0:
                loss_plot_list.append(loss_plot_avg)
                loss_plot = 0.0
                
#             loss_epoch.append(loss)
        
        loss_list[fold] = loss/(len(trainloader))

        show_plot(loss_plot_list, args.plot_every, fold, save_path=save_result_path, file_name=result_img_name)
        print('-----------------------------------')
        print(f'Fold {k_folds} Training Loss: {loss_list[fold]}')
        print('Average Training Loss: %.4f' % (sum(loss_list.values())/len(loss_list.items())))
        
        # Save model
        save_path =  save_model_path + f'/model_fold{fold}.pth'
        torch.save(model.state_dict(), save_path)
        val_loss = 0.0
#         print('Saving model at %s' % save_path)
        
        print('-------- Starting testing --------')
        
        with torch.no_grad():
                        
            for i, data in enumerate(testloader, 0):
                inputs, targets = data
                outputs = model(inputs.to(device))
                outputs = outputs.view(-1).to(device)
                targets = targets.type(torch.FloatTensor).view(-1).to(device)
                losses = loss_function(outputs, targets)
                val_loss += losses.item()
                
            print('val_loss of fold: %.4f' % (val_loss/len(testloader)))
            print('-----------------------------------')
            results[fold] = val_loss/len(testloader)
        
        print(f'K-Fold CV Results of {k_folds} Folds')
        print('-----------------------------------')
        sum_v = 0.0
        for key, value in results.items():
            print('Fold %d: %.4f' % (key, value))
            sum_v += value
        print('Average: %.4f' % (sum_v/len(results.items())))
                  