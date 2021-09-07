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

from src.model import *
from src.data import *
from src.plot_utils import show_plot

import argparse
import datetime
from os import makedirs


PAD_token = 0


parser = argparse.ArgumentParser(description='Parser for training.')
parser.add_argument('-k', '--k_folds', type=int, default=5,
                    help='number of folds for cross validation (default: 5)')
parser.add_argument('-e', '--num_epochs', type=int, default=300,
                    help='number of epochs (default: 500)')
parser.add_argument('-b', '--batch_size', type=int, default=10,
                    help='size of batch (default: 10)')
parser.add_argument('-l', '--learning_rate', type=float, default=1e-5,
                    help='learning rate (default: 1e-5)')
parser.add_argument('-t', '--teacher_forcing_rate', type=float, default=0.5,
                    help='teacher forcing rate (default: 0.5)')
parser.add_argument('-d', '--dropout', type=float, default=0.5,
                    help='encoder dropout rate (default: 0.5)')
parser.add_argument('--embedding_dim', type=int, default=24,
                    help='embedding dimension (default: 24)')
parser.add_argument('--hidden_dim', type=int, default=64,
                    help='hidden dimension (default: 64)')
parser.add_argument('--plot_every', type=int, default=50,
                    help='number of epochs for plotting (default: 50)')
parser.add_argument('--data_path', default='./data/train_df.csv',
                    help='path for train dataframe (default: ./data/train_df.csv')
parser.add_argument('--word_dict', default='./data/word_dict.pkl',
                    help='path for word dict (default: ./data/word_dict.pkl)')
parser.add_argument('--debug', type=bool, default=False,
                    help='debug mode (default: False)')

args = parser.parse_args()
cur_date = datetime.datetime.now().strftime('%y%m%d-%H%M%S')


if __name__ == '__main__':
    gpu = 0
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    k_folds = args.k_folds
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    plot_every = args.plot_every
    emb_dim = args.embedding_dim
    hidn_dim = args.hidden_dim
    
    result_img_name = 'train'
    save_model_path = f'./model/AE/{cur_date}/'
    save_result_path = f'./result/AE/{cur_date}/'

    loss_function = nn.CrossEntropyLoss(ignore_index=PAD_token)
    results = {}
    
    torch.manual_seed(42)
    
    dataset_train = pd.read_csv(args.data_path).to_numpy()
    with open(args.word_dict,'rb') as f:
        word2index_dict = pickle.load(f)
    vocab_size = len(word2index_dict)
    
    dataset = Dataset(dataset_train, word2index_dict)
    kfold = KFold(n_splits=k_folds, shuffle=True)
    import pdb; pdb.set_trace()     
    # Make directory for saved model
    try:
        makedirs(save_model_path, exist_ok=True)
        makedirs(save_result_path, exist_ok=True)
    except:
        save_model_path = './model/'
        save_result_path = './result/'
        
    print('----------------------------------')
    
    for fold, (train_index, test_index) in enumerate(kfold.split(dataset)):
        
        print(f'Fold {fold}')
        print('----------------------------------')

        loss_plot_list = []
        loss_plot = 0.0
    
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_index)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_index)
    
        trainloader = get_loader(dataset, train_subsampler, batch_size)
        testloader = get_loader(dataset, test_subsampler, batch_size)
        
        encoder = Encoder(vocab_size, emb_dim, hidn_dim, device, args.dropout).to(device)
        decoder = Decoder(vocab_size, emb_dim, hidn_dim, device).to(device)
        model = Seq2Seq(encoder, decoder, device).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        for epoch in range(1, num_epochs+1):
            loss = 0.0
                  
            for i, data in enumerate(trainloader, 0):
                inputs, _, targets, _ = data
                outputs, _ = model(inputs.to(device), targets.to(device), args.teacher_forcing_rate)
                output_dim = outputs.size(-1)
                outputs = outputs[:, 1:].contiguous().view(-1, output_dim)
                targets = targets[:, 1:].contiguous().view(-1).to(device)
                
                losses = loss_function(outputs, targets)
                  
                losses.backward()
                optimizer.step()
                  
                loss += losses.item()

                loss_plot += losses.item()
                
            if epoch % plot_every == 0:
                print('Epoch %d / %d (%d%%) Loss: %.4f' % (epoch, num_epochs, epoch / num_epochs * 100, loss_plot / (plot_every*(i+1))))
                loss_plot_list.append(loss_plot / (plot_every*len(trainloader)))
                loss_plot = 0.0

        show_plot(loss_plot_list, plot_every, fold, save_path=save_result_path, file_name=result_img_name)
        
        print('Training has finished. Saving trained model.')
        
        print('Starting testing')
        
        # Save model
        save_path =  save_model_path + f'/model_fold{fold}.pth'
        save_path_encoder = save_model_path + f'./encoder_fold{fold}.pth'
        torch.save(model.state_dict(), save_path)
        torch.save(encoder.state_dict(), save_path_encoder)
        val_loss = 0.0
        
        with torch.no_grad():
                        
            for i, data in enumerate(testloader, 0):
                inputs, _, targets, _ = data
                outputs, _ = model(inputs, targets)
                output_dim = outputs.size(-1)
                outputs = outputs[:, 1:].contiguous().view(-1, output_dim)
                targets = targets[:, 1:].contiguous().view(-1).to(device)
                losses = loss_function(outputs, targets)
                val_loss += losses.item()
                
            print('val_loss of fold: %.4f' % (val_loss/len(testloader)))
            print('-----------------------------------')
            results[fold] = val_loss/len(testloader)
        
        print(f'K-Fold CV Results of {k_folds} Folds')
        print('-----------------------------------')
        sum = 0.0
        for key, value in results.items():
            print('Fold %d: %.4f' % (key, value))
            sum += value
        print('Average: %.4f' % (sum/len(results.items())))
                  