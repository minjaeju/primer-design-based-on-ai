import numpy as np
import torch
import torch.utils.data as data


class Dataset(data.Dataset):

    def __init__(self, seq, dic):
        
        self.src_seqs = seq
        self.trg_seqs = seq
        self.num_total_seqs = len(self.src_seqs)
        self.word2id = dic
        import pdb; pdb.set_trace()
    def __getitem__(self, index):

        src_seq = self.src_seqs[index][0]
        trg_seq = self.trg_seqs[index][0]
        src_seq = self.preprocess(src_seq, self.word2id)
        trg_seq = self.preprocess(trg_seq, self.word2id)
        import pdb; pdb.set_trace()
        return src_seq, trg_seq

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, dictionary, trg=True):

        seq2num = []
        seq2num.append(dictionary['SOS'])
        seq2num.extend([dictionary[nt] for nt in sequence if nt in dictionary])
        seq2num.append(dictionary['EOS'])
        seq2num = torch.Tensor(seq2num)
        #
        return seq2num
    
class Dataset_Reg(data.Dataset):

    def __init__(self, data, dic):
        
        self.data = data
        self.num_total_seqs = len(self.data)
        self.word2id = dic

    def __getitem__(self, index):

        seq = self.data[index][0]
        target = self.data[index][1]
        seq = self.preprocess(seq, self.word2id)
        target = torch.Tensor(np.array([target]))
        return seq, target

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, dictionary, trg=True):
        
        seq2num = []
        seq2num.append(dictionary['SOS'])
        seq2num.extend([dictionary[nt] for nt in sequence if nt in dictionary])
        seq2num.append(dictionary['EOS'])
        seq2num = torch.Tensor(seq2num)
        return seq2num
    
class Dataset_infer(data.Dataset):

    def __init__(self, data, dic):
        
        self.data = data
        self.num_total_seqs = len(self.data)
        self.word2id = dic

    def __getitem__(self, index):

        seq = self.data[index][0]
        seq = self.preprocess(seq, self.word2id)
        return seq

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, dictionary, trg=True):
        
        seq2num = []
        seq2num.append(dictionary['SOS'])
        seq2num.extend([dictionary[nt] for nt in sequence if nt in dictionary])
        seq2num.append(dictionary['EOS'])
        seq2num = torch.Tensor(seq2num)
        return seq2num

def collate_fn(data):
    
    data.sort(key=lambda x: len(x[0]), reverse=True)

    src_seqs, trg_seqs = zip(*data)

    src_seqs, src_lengths = merge(src_seqs)
    trg_seqs, trg_lengths = merge(trg_seqs)
    
    return src_seqs, src_lengths, trg_seqs, trg_lengths

def collate_reg_fn(data):
    
    data.sort(key=lambda x: len(x[0]), reverse=True)

    seqs, targets = zip(*data)

    pad_seq, seq_length = merge(seqs)
    target = merge_reg(targets)
    return  pad_seq, seq_length, target

def collate_infer_fn(data):
    
    data.sort(key=lambda x: len(x), reverse=True)

    seq= data
    
    pad_seq, seq_length = merge(seq)
        
    return pad_seq, seq_length

def merge(sequences):
    lengths = [len(seq) for seq in sequences]
    padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
    for i, seq in enumerate(sequences):
        end = lengths[i]
        padded_seqs[i, :end] = seq[:end]
    return padded_seqs, lengths

def merge_reg(targets):
    lengths = [len(target) for target in targets]
    target_merge = torch.zeros(len(targets), max(lengths)).float()
    for i, t in enumerate(targets):
        end = lengths[i]
        target_merge[i, :end] = t[:end]
    return target_merge

def get_loader(dataset, sampler, batch_size):

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              collate_fn=collate_fn,
                                              sampler=sampler)

    return data_loader

def get_loader_reg(dataset, sampler, batch_size):

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              collate_fn=collate_reg_fn,
                                              sampler=sampler)

    return data_loader

def get_loader_infer(dataset, batch_size):

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              collate_fn=collate_infer_fn)

    return data_loader

class Dataset_FRP(data.Dataset):
    
    def __init__(self, data, key):
        
        self.data = data
        self.key = key
        self.num_total_seqs = len(self.data)
        self.max_seq_len = 0
    
    def __getitem__(self, index):
        #Fprimer:0, probe:1, Rprimer:2
        seqs = self.data.iloc[index][self.key]
        target = self.data.iloc[index]['ct']
        target = torch.Tensor(np.array([target]))

        return seqs, target

    def __len__(self):
        return self.num_total_seqs
    
    def set_max_seq_len(self, max_seq_len=None):
        if max_seq_len is None:
            self.max_seq_len = max([len(i) for i in self.data[self.key]])
        else:
            self.max_seq_len = max_seq_len

class CollateCNN:
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

        data.sort(key=lambda x: len(x[0]), reverse=True)

        if self.is_test:
            seqs, _ = zip(*data)
            onehot = self.one_hot_encode(seqs)
            return onehot
            
        else:
            seqs, targets = zip(*data)
            onehot = self.one_hot_encode(seqs)
            tars = merge_reg(targets)

            return onehot, tars
    
    def one_hot_encode(self, seqs):
        
        if self.key == 'R primer':
            seqs = list(seqs)
            for i in list(range(len(seqs))):
                seqs[i] = self.comp(seqs[i])
            seqs = tuple(seqs)

        lengths = [len(seq) for seq in seqs]
        base_dict = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'R': [0,2], 'Y': [1,3], 'M': [0,3], 'K': [1,2]}
        onehot = torch.zeros(len(seqs), len(self.word2id), 40).float()
        
        for i, s in enumerate(seqs):
            for j, base in enumerate(s):
                index = i, base_dict[base],j
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
    

def get_loader_CNN(dataset, subsampler, batch_size, key, word2dict:dict, is_test=False):

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              sampler=subsampler,
                                              collate_fn=CollateCNN(dataset.max_seq_len, key, word2dict, is_test))
    
    return data_loader
    
    
def get_loader_CNN_infer(dataset, batch_size, key, word2dict:dict, is_test=True):

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              collate_fn=CollateCNN(dataset.max_seq_len, key, word2dict, is_test))
    
    return data_loader

