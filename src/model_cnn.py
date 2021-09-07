import torch
import torch.nn as nn
import torch.nn.functional as F

import random

class CNN(nn.Module):
    def __init__(self, input_dim, vocab_size, hidn_dim, device, kernel_size=3, dropout=0.3):
        super().__init__()
        self.device = device
        self.input_dim = input_dim   
        self.vocab_size = vocab_size
        self.hidn_dim = hidn_dim      
        
        self.conv = nn.Conv1d(vocab_size, vocab_size, kernel_size, padding=(kernel_size-1)//2) 
        self.maxpool = nn.MaxPool1d(5) 
        self.dense_hidn = nn.Linear(vocab_size * (input_dim//4), hidn_dim)

        self.dense_out = nn.Linear(hidn_dim, 1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs):
        """ 
        inputs: sequence one-hot encoding
            - (batch_size, vocab_size, max_seq_len)
        targets: RFU, CT
            - (batch_size, 1)
        """

        output = F.relu(self.conv(inputs))
        output = self.maxpool(output)
        output = self.flatten(output)
        output = self.dropout(self.dense_hidn(output))
        output = self.dense_out(output)

        return output


class MultiInputCNN(nn.Module):
    def __init__(self, input_dim, vocab_size, hidn_dim, device, kernel_size=3, dropout=0.3):
        super().__init__()
        self.device = device
        self.input_dim = input_dim    
        self.vocab_size = vocab_size
        self.hidn_dim = hidn_dim      
        self.pool_size = 4           
        
        kernel_1 = 3
        kernel_2 = 5
        kernel_3 = 7
        # same padding
        self.conv_f_1 = nn.Conv1d(vocab_size, vocab_size, kernel_1, padding=(kernel_1-1)//2)
        self.conv_f_2 = nn.Conv1d(vocab_size, vocab_size, kernel_2, padding=(kernel_2-1)//2)
        self.conv_f_3 = nn.Conv1d(vocab_size, vocab_size, kernel_3, padding=(kernel_3-1)//2)
        self.conv_p_1 = nn.Conv1d(vocab_size, vocab_size, kernel_1, padding=(kernel_1-1)//2)
        self.conv_p_2 = nn.Conv1d(vocab_size, vocab_size, kernel_2, padding=(kernel_2-1)//2)
        self.conv_p_3 = nn.Conv1d(vocab_size, vocab_size, kernel_3, padding=(kernel_3-1)//2)
        self.conv_r_1 = nn.Conv1d(vocab_size, vocab_size, kernel_1, padding=(kernel_1-1)//2)
        self.conv_r_2 = nn.Conv1d(vocab_size, vocab_size, kernel_2, padding=(kernel_2-1)//2)
        self.conv_r_3 = nn.Conv1d(vocab_size, vocab_size, kernel_3, padding=(kernel_3-1)//2)
        
        self.maxpool_f = nn.MaxPool1d(self.pool_size)
        self.maxpool_p = nn.MaxPool1d(self.pool_size)
        self.maxpool_r = nn.MaxPool1d(self.pool_size)
        
        self.dense_hidn = nn.Linear(3 * vocab_size*2 * input_dim, hidn_dim)
        self.dense_hidn_species = nn.Linear(3 * vocab_size*2 * input_dim +2, hidn_dim)  # input_dim//input_dim=1

        self.dense_out = nn.Linear(hidn_dim, 1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs, species=None):
        """ 
        inputs: sequence one-hot encoding
            - (batch_size, vocab_size, max_seq_len)
        targets: RFU, CT
            - (batch_size, 1)
        """
        fprimer = inputs[0].to(self.device)
        #probe = inputs[1].to(self.device)
        rprimer = inputs[1].to(self.device)
        # Model 1
        fprimer_output_1 = F.relu(self.conv_f_1(fprimer))
        fprimer_output_2 = F.relu(self.conv_f_2(fprimer))
        fprimer_output_3 = F.relu(self.conv_f_3(fprimer))
        fprimer_output = torch.cat((fprimer_output_1, fprimer_output_2, fprimer_output_3), 1)
        fprimer_output = self.flatten(fprimer_output)
        """
        # Model 2
        probe_output_1 = F.relu(self.conv_p_1(probe))
        probe_output_2 = F.relu(self.conv_p_2(probe))
        probe_output_3 = F.relu(self.conv_p_3(probe))
        probe_output = torch.cat((probe_output_1, probe_output_2, probe_output_3), 1)
        probe_output = self.maxpool_p(probe_output)
        probe_output = self.flatten(probe_output)
        """
        # Model 3
        rprimer_output_1 = F.relu(self.conv_r_1(rprimer))
        rprimer_output_2 = F.relu(self.conv_r_2(rprimer))
        rprimer_output_3 = F.relu(self.conv_r_3(rprimer))
        rprimer_output = torch.cat((rprimer_output_1, rprimer_output_2, rprimer_output_3), 1)
        rprimer_output = self.flatten(rprimer_output)
        output = torch.cat((fprimer_output, rprimer_output), 1)
        if species is not None:
            species = species.to(self.device)
            output = torch.cat((output, species),1)
            output = self.dropout(self.dense_hidn_species(output))    
                
        else:
            print(output.shape)
            output = self.dropout(self.dense_hidn(output))
        output = self.dense_out(output)
        
        return output