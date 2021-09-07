import torch
import torch.nn as nn
from torch import optim
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import KFold
from src.model_cnn import *
from src.data_multi_input import *

import argparse
import datetime
from os import listdir
from os.path import isdir, join, basename, dirname

import pdb


parser = argparse.ArgumentParser(description='Parser for inference.')
parser.add_argument('-k', '--k_folds', type=int, default=5,
                    help='number of folds for cross validation (default: 5)')
parser.add_argument('-b', '--batch_size', type=int, default=4,
                    help='size of batch (default: 20)')
parser.add_argument('--species', type=bool, default=True,
                    help='add species feature (default: True)')
parser.add_argument('--hidden_dim', type=int, default=64,
                    help='hidden dimension (default: 64)')
parser.add_argument('--data_path', default='./data/train_210729+210820_drop.csv',
                    help='path for train dataframe (default: ./data/210729_drop.csv')
parser.add_argument('--key', default='F primer',
                    help='sequence type for prediction (default: F primer)')
parser.add_argument('--model_path', 
                    help='path for trained encoder (default: latest model)')
parser.add_argument('--word_dict', default='./data/word_dict.pkl',
                    help='path for word dict (default: ./data/word_dict.pkl)')
parser.add_argument('--kernel_size', type=int, default=3,
                    help='conv kernel size (default: 3)')

parser.add_argument('--max_len', type=int, default=40,
                    help='max sequence length (default: 40)')

args = parser.parse_args()


if __name__ == '__main__':
    gpu = 0
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    
    k_folds = args.k_folds
    batch_size = args.batch_size
    hidn_dim = args.hidden_dim
    key = args.key
    spec = args.species
    model_root_path = './model/cnn'
    result_name = f'./result/result.csv'
    
    # Load recent model
    load_model_path = args.model_path
    if load_model_path is None:
        model_dirs = [d for d in listdir(model_root_path) if isdir(join(model_root_path, d))]
        model_dirs.sort()
        cur_date = model_dirs[-1]
        load_model_path = f'{model_root_path}/{cur_date}/'
        result_name = f'./result/result_{cur_date}.csv'
    else:
        cur_date = load_model_path.split('/')[-1]
        result_name = f'./result/result_{cur_date}.csv'
        
    print(f'Load cnn model at {load_model_path}')
    
    torch.manual_seed(42)
    
    dataset_df = pd.read_csv(args.data_path)

    word2index_dict = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    vocab_size = len(word2index_dict)
    
    dataset = Dataset_FRP(dataset_df, key)
    dataset.set_max_seq_len(args.max_len)
    testloader = get_loader_CNN_infer(dataset, batch_size, key, word2index_dict, is_test=True)
    result = []
    
    for fold in list(range(k_folds)):
        model = MultiInputCNN(args.max_len, vocab_size, hidn_dim, device, args.kernel_size).to(device)
        model.eval()
        path_cnn = join(load_model_path, f'model_fold{fold}.pth')
        
        fold_result = []
        try:
            checkpoint_cnn = torch.load(path_cnn)
            
            model.load_state_dict(checkpoint_cnn)

            for i, data in enumerate(testloader, 0):
                inputs, species = data
                if spec == True:
                    outputs = model(inputs, species)
                else:
                    outputs = model(inputs)
                output_cpu = outputs.detach().cpu().numpy()
                fold_result += list(output_cpu)

        except Exception as e:
            print(e)
        
        result.append(fold_result)
    
    if len(result) == 0:
        print(f'No model exists')
    else:     
        mean = np.mean(result, axis=0)
        print(np.squeeze(mean))
        dataset_df['ct_pred'] = mean
        
        dataset_df.to_csv(result_name, index=0)
    
    print('Inference Completed')
