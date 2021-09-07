import numpy as np
import torch
import torch.utils.data as data

import pdb


def merge_reg(targets, species):
    lengths = [len(target) for target in targets]
    target_merge = torch.zeros(len(targets), max(lengths)).float()
    species_merge = torch.zeros(len(targets),2).float()
    for i, t in enumerate(targets):
        end = lengths[i]
        target_merge[i, :end] = t[:end]
    for i, s in enumerate(species):
        if s == 'pyogenes':
            species_merge[i][0] = 1
        else:
            species_merge[i][1] = 1
    return target_merge, species_merge

class Dataset_FRP(data.Dataset):
    
    def __init__(self, data, key):
        
        self.data = data
        self.key = key
        self.num_total_seqs = len(self.data)
        self.max_seq_len = 0
    
    def __getitem__(self, index):
        #Fprimer:f/0, probe:b/1, Rprimer:r/2
        fprimer = self.data.iloc[index]['F primer']
        #probe = self.data.iloc[index]['probe']
        rprimer = self.data.iloc[index]['R primer']
        #prd_seq = self.data.iloc[index]['Product Sequence']
        species = self.data.iloc[index]['species']
        target = self.data.iloc[index]['ct']
        target = torch.Tensor(np.array([target]))

        return fprimer, rprimer, target, species

    def __len__(self):
        return self.num_total_seqs
    
    def set_max_seq_len(self, max_seq_len=None):
        if max_seq_len is None:
            self.max_seq_len = max([len(i) for i in self.data[self.key]])
        else:
            self.max_seq_len = max_seq_len

    
class CollateMultiCNN:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """
    def __init__(self, max_seq_len, key, word2index:dict, is_test=False):
        self.max_seq_len = max_seq_len
        self.word2id = word2index
        self.is_test = is_test
        self.key = key
        
    def collate_CNN_fn(self, data):
        # one hot -> list (to cover multiple columns)
        # DO NOT SORT or SHUFFLE
        fprimer, rprimer, targets, species = zip(*data)
        fprimer = self.one_hot_encode(fprimer, self.max_seq_len, self.word2id)
        #probe = self.one_hot_encode(probe, self.max_seq_len, self.word2id)
        rprimer = self.one_hot_encode(rprimer, self.max_seq_len, self.word2id, key='R primer')
            
        tars, species = merge_reg(targets, species)
        if self.is_test:
            return (fprimer, rprimer), species
        else:
            return (fprimer, rprimer), tars, species
            
    def one_hot_encode(self, seqs, max_seq_len, word2id, key=None):
        # use special base dict for R/F primer and probe
        base_dict = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'R': [0,2], 'Y': [1,3], 'M': [0,3], 'K': [1,2]}
       
        if key == 'R primer':
            seqs = list(seqs)
            for i in list(range(len(seqs))):
                seqs[i] = self.comp(seqs[i])
            seqs = tuple(seqs)
        lengths = [len(seq) for seq in seqs]
        onehot = torch.zeros(len(seqs), len(word2id), max_seq_len).float()
        
        for i, s in enumerate(seqs):
            for j, base in enumerate(s):
                try:
                    index = i, base_dict[base],j
                except:
                    print(base)
                onehot[index] = 1
        return onehot
    
    def comp(self, seq):
        compDict = {"A":"T", "G":"C", "T":"A", "C":"G", "Y":"R", "M":"K", "R":"Y", "K":"M"}
        retList = []
        for ele in seq:
            if ele not in compDict:
                continue
            retList.append(compDict[ele])
        
        return "".join(reversed(retList))    
    
    def __call__(self, batch):
        return self.collate_CNN_fn(batch)
    

def get_loader_CNN(dataset, subsampler, batch_size, key, word2dict:dict, max_seq_len=40, is_test=False):

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              sampler=subsampler,
                                              collate_fn=CollateMultiCNN(max_seq_len, key, word2dict, is_test))
    
    return data_loader
    
    
def get_loader_CNN_infer(dataset, batch_size, key, word2dict:dict, max_seq_len=40, is_test=True):

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              collate_fn=CollateMultiCNN(max_seq_len, key, word2dict, is_test))
    
    return data_loader

