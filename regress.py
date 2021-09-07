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
from os import makedirs, listdir
from os.path import isdir, join


parser = argparse.ArgumentParser(description='Parser for regression.')
parser.add_argument('-k', '--k_folds', type=int, default=5,
                    help='number of folds for cross validation (default: 5)')
parser.add_argument('-e', '--num_epochs', type=int, default=250,
                    help='number of epochs (default: 1000)')
parser.add_argument('-b', '--batch_size', type=int, default=10,
                    help='size of batch (default: 10)')
parser.add_argument('-l', '--learning_rate', type=float, default=1e-5,
                    help='learning rate (default: 1e-4)')
parser.add_argument('-d', '--dropout', type=float, default=0.3,
                    help='regression dropout rate (default: 0.3)')
parser.add_argument('--embedding_dim', type=int, default=24,
                    help='embedding dimension (default: 24)')
parser.add_argument('--hidden_dim', type=int, default=64,
                    help='hidden dimension (default: 64)')
parser.add_argument('--plot_every', type=int, default=50,
                    help='number of epochs for plotting (default: 50)')
parser.add_argument('--data_path', default='./data/train_df45.csv',
                    help='path for train dataframe (default: ./data/train_df.csv)')
parser.add_argument('--model_path', 
                    help='path for trained encoder (default: latest model)')
parser.add_argument('--word_dict', default='./data/word_dict.pkl',
                    help='path for word dict (default: ./data/word_dict.pkl)')
parser.add_argument('--debug', type=bool, default=False,
                    help='debug mode (default: False)')

args = parser.parse_args()


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
    
    # File names
    result_img_name = 'regress'
    load_model_path = args.model_path

    # Load recent model
    if load_model_path is None:
        model_dirs = [d for d in listdir('./model/AE') if isdir(join('./model/AE', d))]
        model_dirs.sort()
        cur_date = model_dirs[-1]
        load_model_path = f'./model/AE/{cur_date}/'
        save_model_path = f'./model/reg/{cur_date}/'
        save_result_path = f'./result/reg/{cur_date}/'
    else:
        save_model_path = load_model_path.replace('AE', 'reg')
        save_result_path = save_model_path.replace('model', 'result')
        
    print(f'Load encoder model at {load_model_path}')
    
    #loss_function = nn.L1Loss()
    loss_function = nn.MSELoss()
    results = {}
    
    torch.manual_seed(42)
    
    dataset_train = pd.read_csv(args.data_path).to_numpy()
    with open(args.word_dict,'rb') as f:
        word2index_dict = pickle.load(f)
    
    dataset = Dataset_Reg(dataset_train, word2index_dict)
    kfold = KFold(n_splits=k_folds, shuffle=True)
    
    # Make directory for saved model
    try:
        makedirs(save_model_path, exist_ok=True)
        makedirs(save_result_path, exist_ok=True)
    except:
        save_model_path = './model/reg/'
        save_result_path = './result/reg/'
    
    print('----------------------------------')
    
    for fold, (train_index, test_index) in enumerate(kfold.split(dataset)):
        
        print(f'Fold {fold}')
        print('----------------------------------')
    
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_index)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_index)
    
        trainloader = get_loader_reg(dataset, train_subsampler, batch_size)
        testloader = get_loader_reg(dataset, test_subsampler, batch_size)
        
        encoder = Encoder(7, emb_dim, hidn_dim, device).to(device)
        path = join(load_model_path, f'encoder_fold{fold}.pth')
        checkpoint = torch.load(path)
        encoder.load_state_dict(checkpoint)
        model = Regressor(hidn_dim, hidn_dim, device, args.dropout).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # parameters for printing
        loss_plot_list = []
        loss_plot = 0.0
        
        for epoch in range(1, num_epochs + 1):
            loss = 0.0
            
            for i, data in enumerate(trainloader, 0):
                inputs, _, targets = data
                enc_hidn = torch.zeros(1, inputs.shape[0], hidn_dim, device=device)
                enc_cell = torch.zeros(1, inputs.shape[0], hidn_dim, device=device)
                outputs, _, _ = encoder(inputs, enc_hidn, enc_cell)
                output = model(outputs)

                losses = loss_function(output, targets.to(device))
                
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
        save_path = join(save_model_path, f'reg_fold{fold}.pth')
        torch.save(model.state_dict(), save_path)
        
        val_loss = 0.0
        with torch.no_grad():
                        
            for i, data in enumerate(testloader, 0):
                inputs, _, targets = data
                enc_hidn = torch.zeros(1, inputs.shape[0], 64, device=device)
                enc_cell = torch.zeros(1, inputs.shape[0], 64, device=device)               
                outputs, _, _= encoder(inputs,enc_hidn,enc_cell)
                output = model(outputs)
                losses = loss_function(output, targets.to(device))
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