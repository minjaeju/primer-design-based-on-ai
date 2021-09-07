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
from src.data_multi_input import *
from src.plot_utils import show_plot

import argparse
import datetime
from os import makedirs

from src.callbacks import EarlyStopping
# check output
import scipy
import sklearn
import pdb 


parser = argparse.ArgumentParser(description='Parser for training.')
parser.add_argument('-k', '--k_folds', type=int, default=5,
                    help='number of folds for cross validation (default: 5)')
parser.add_argument('-e', '--num_epochs', type=int, default=3000,
                    help='number of epochs (default: 500)')
parser.add_argument('-b', '--batch_size', type=int, default=4,
                    help='size of batch (default: 10)')
parser.add_argument('-l', '--learning_rate', type=float, default=1e-5,
                    help='learning rate (default: 1e-5)')
# parser.add_argument('-t', '--teacher_forcing_rate', type=float, default=0.5,
#                     help='teacher forcing rate (default: 0.5)')
parser.add_argument('-d', '--dropout', type=float, default=0.3,
                    help='encoder dropout rate (default: 0.5)')
# parser.add_argument('--embedding_dim', type=int, default=24,
#                     help='embedding dimension (default: 24)')
parser.add_argument('--species', default=True,
                    help='add species feature (default: True)')
parser.add_argument('--hidden_dim', type=int, default=512,
                    help='hidden dimension (default: 64)')
parser.add_argument('--plot_every', type=int, default=10,
                    help='number of epochs for plotting (default: 10)')
parser.add_argument('--print_every', type=int, default=1,
                    help='number of epochs for printing losses for plot (default: 1)')
parser.add_argument('--data_path', default='./data/train_210729+210820_drop.csv',
                    help='path for train dataframe (default: ./data/train_df.csv')
parser.add_argument('--key', default='R primer',
                    help='sequence type for prediction (default: R primer)')
parser.add_argument('--word_dict', default='./data/word_dict.pkl',
                    help='path for word dict (default: ./data/word_dict.pkl)')
parser.add_argument('--kernel_size', type=int, default=3,
                    help='conv kernel size (default: 3)')
# parser.add_argument('--debug', type=bool, default=False,
#                     help='debug mode (default: False)')
parser.add_argument('--gpu', type=int, default=1,
                    help='gpu assigned (default: 0)')
parser.add_argument('--patience', type=int, default=20,
                    help='gpu assigned (default: 20)')

args = parser.parse_args()
cur_date = datetime.datetime.now().strftime('%y%m%d-%H%M%S')


if __name__ == '__main__':
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    k_folds = args.k_folds
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
#     plot_every = args.plot_every
#     emb_dim = args.embedding_dim
    hidn_dim = args.hidden_dim
    key = args.key
    spec = args.species
    max_seq_len = 40
    
    result_img_name = 'train'
    save_model_path = f'./model/cnn/{cur_date}/'
    save_result_path = f'./result/cnn/{cur_date}/'
    
    # Experiment results (loss)
    train_results = {}
    val_results = {}
    
    torch.manual_seed(42)
    
    # Load datasets
    dataset_train = pd.read_csv(args.data_path)  #.to_numpy()
#     with open(args.word_dict,'rb') as f:
#         word2index_dict = pickle.load(f)
#     word2index_dict = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'R':4, 'Y':5, 'M':6, 'K':7}
    word2index_dict = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    vocab_size = len(word2index_dict)
    
    dataset = Dataset_FRP(dataset_train, key)
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
        
        save_path = save_model_path + f'/model_fold{fold}.pth'
        # parameters for printing
        train_loss_plot_list = []
        train_loss_plot = 0.0
        # avg losses per epoch
        avg_train_losses = []
        avg_val_losses = []
        # correlation
        corr_plot_list = []
        # early stop
        early_stop_pat = args.patience
        early_stop_cnt = 0
        best_train_loss = np.inf
        best_valid_loss = np.inf

        # data
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_index)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_index)
    
        trainloader = get_loader_CNN(dataset, train_subsampler, batch_size, key, word2index_dict)
        testloader = get_loader_CNN(dataset, test_subsampler, batch_size, key, word2index_dict)
        
        # model
        model = MultiInputCNN(max_seq_len, vocab_size, hidn_dim, device, args.kernel_size, args.dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_function = nn.MSELoss()

        for epoch in range(1, num_epochs+1):
            
            # losses per epoch while training/testing
            train_loss = 0.0
            valid_loss = 0.0
            train_losses = []
            valid_losses = []
            # correlation
            corr_list = []
            
            # train
            model.train()

            for i, data in enumerate(trainloader, 0):
                inputs, targets, species = data
                if spec == True:
                    outputs = model(inputs, species)
                else:
                    outputs = model(inputs)
                outputs = outputs.view(-1).to(device)
                targets = targets.type(torch.FloatTensor).view(-1).to(device)                
                # print(outputs.size())
                # print(outputs)
                losses = loss_function(outputs, targets)
                  
                losses.backward()
                optimizer.step()
                
                train_loss += losses.item()
                train_loss_plot += losses.item()
                train_losses.append(losses.item())

                # pdb.set_trace()
                if len(outputs) >= 2:
                    cor, p = scipy.stats.pearsonr(list(outputs.cpu().detach().numpy()), list(targets.cpu().detach().numpy()))
                    corr_list.append(cor)

            # evaluate
            model.eval()

            for i, data in enumerate(testloader, 0):
                inputs, targets, species = data
                if spec == True:
                    outputs = model(inputs, species)
                else:
                    outputs = model(inputs)
                outputs = outputs.view(-1).to(device)
                targets = targets.type(torch.FloatTensor).view(-1).to(device)
                
                losses = loss_function(outputs, targets)

                valid_loss += losses.item()
                valid_losses.append(losses.item())

            # avg(loss) per epoch
            avg_train_loss = np.average(train_losses)
            avg_train_losses.append(avg_train_loss)
            avg_val_loss = np.average(valid_losses)
            # early stop
            if len(avg_val_losses) != 0 and avg_val_loss >= avg_val_losses[-1]:
                early_stop_cnt += 1
            else:
                early_stop_cnt = 0
                best_train_loss = avg_train_loss
                best_valid_loss = avg_val_loss
                torch.save(model.state_dict(), save_path)
            avg_val_losses.append(avg_val_loss)
            
            if early_stop_cnt >= early_stop_pat:
                break
            
            # Plot
            avg_loss_plot = train_loss_plot / (args.plot_every*(i+1))
            avg_corr = np.average(corr_list)
            corr_plot_list.append(avg_corr)
            if epoch % args.print_every == 0:
                print('Epoch %d / %d (%d%%) train loss: %.4f, valid loss: %.4f, correlation: %.4f' \
                    % (epoch, num_epochs, epoch / num_epochs * 100, avg_train_loss, avg_val_loss, avg_corr))

            if epoch % args.plot_every == 0:
                train_loss_plot_list.append(avg_loss_plot)
                train_loss_plot = 0.0
        
        # train_results[fold] = train_loss/(len(trainloader))
        train_results[fold] = best_train_loss
        val_results[fold] = best_valid_loss
        
        show_plot(train_loss_plot_list, args.plot_every, fold, save_path=save_result_path, file_name=result_img_name)
        
        print('-----------------------------------')
        print(f'Fold {fold} Training Loss: {train_results[fold]}')
        print('Average Training Loss: %.4f' % (sum(train_results.values())/len(train_results.items())))
        
        # Save model
        # save_path = save_model_path + f'/model_fold{fold}.pth'
        # torch.save(model.state_dict(), save_path)
        checkpoint_cnn = torch.load(save_path)
        model.load_state_dict(checkpoint_cnn)
        
        print('-------- Starting testing --------')
        
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
                        
            for i, data in enumerate(testloader, 0):
                inputs, targets, species = data
                if spec == True:
                    outputs = model(inputs, species)
                else:
                    outputs = model(inputs)
                outputs = outputs.view(-1).to(device)
                targets = targets.type(torch.FloatTensor).view(-1).to(device)
                losses = loss_function(outputs, targets)
                val_loss += losses.item()
                
            print('val_loss of fold: %.4f' % (val_loss/len(testloader)))
            print('-----------------------------------')
            val_results[fold] = val_loss/len(testloader)
        
        print(f'K-Fold CV Results of {k_folds} Folds')
        print('-----------------------------------')
        for f in range(fold+1):
            print('[Fold %d] train loss: %.4f, valid loss: %.4f' %\
                (f, train_results[f], val_results[f]))
        print('%d folds average train loss: %.4f, valid loss: %.4f' \
            % ((fold+1), sum(train_results.values())/len(train_results.items()),
               sum(val_results.values())/len(val_results.items())))
        print('Saved Model Path: %s' % save_model_path)
        print('-----------------------------------')