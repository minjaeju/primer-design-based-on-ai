import torch
import torch.nn as nn
from torch import optim
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import KFold

from src.model import *
from src.data import *

import argparse
import datetime
from os import listdir
from os.path import isdir, join


parser = argparse.ArgumentParser(description='Parser for inference.')
parser.add_argument('-k', '--k_folds', type=int, default=5,
                    help='number of folds for cross validation (default: 5)')
parser.add_argument('-b', '--batch_size', type=int, default=20,
                    help='size of batch (default: 4)')
parser.add_argument('--embedding_dim', type=int, default=24,
                    help='embedding dimension (default: 24)')
parser.add_argument('--hidden_dim', type=int, default=64,
                    help='hidden dimension (default: 64)')
parser.add_argument('--data_path', default='./data/test_df.csv',
                    help='path for train dataframe (default: ./data/test_df.csv')
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
    batch_size = args.batch_size
    emb_dim = args.embedding_dim
    hidn_dim = args.hidden_dim 
    
    load_model_path = args.model_path
    result_name = f'./result/result.csv'

    # Load recent model
    if load_model_path is None:
        model_dirs = [d for d in listdir('./model/AE') if isdir(join('./model/AE', d))]
        model_dirs.sort()
        cur_date = model_dirs[-1]
        load_model_path = f'./model/AE/{cur_date}/'
        load_reg_path = f'./model/reg/{cur_date}/'
        result_name = f'./result/result_{cur_date}.csv'
    else:
        cur_date = load_model_path.split('/')[-1]
        load_reg_path = load_model_path.replace('AE', 'reg')
        result_name = f'./result/result_{cur_date}.csv'
        
    print(f'Load encoder model at {load_model_path}')
    print(f'Load regression model at {load_reg_path}')
    
    torch.manual_seed(42)
    
    dataset_df = pd.read_csv(args.data_path)
    dataset_test = dataset_df.to_numpy()
    with open(args.word_dict,'rb') as f:
        word2index_dict = pickle.load(f)
    vocab_size = len(word2index_dict)
    
    dataset = Dataset_infer(dataset_test, word2index_dict)
    testloader = get_loader_infer(dataset, batch_size = batch_size)
    result = []
    
    for fold in list(range(k_folds)):
        encoder = Encoder(vocab_size, emb_dim, hidn_dim, device).to(device)
        model = Regressor(hidn_dim, hidn_dim, device).to(device)
        path_enc = join(load_model_path, f'encoder_fold{fold}.pth')
        path_reg = join(load_reg_path, f'reg_fold{fold}.pth')
        checkpoint_enc = torch.load(path_enc)
        checkpoint_reg = torch.load(path_reg)
        encoder.load_state_dict(checkpoint_enc)
        model.load_state_dict(checkpoint_reg)

        for i, data in enumerate(testloader, 0):
            inputs, _= data
            enc_hidn = torch.zeros(1, inputs.shape[0], hidn_dim, device=device)
            enc_cell = torch.zeros(1, inputs.shape[0], hidn_dim, device=device)
            outputs, _, _ = encoder(inputs,enc_hidn,enc_cell)
            output = model(outputs)
            output_cpu = output.detach().cpu().numpy()
            result.append(output_cpu)
    mean = np.mean(result, axis=0)
    dataset_df['ct'] = mean
    
    dataset_df.to_csv(result_name, index=0)
    
    print('Inference Completed')
