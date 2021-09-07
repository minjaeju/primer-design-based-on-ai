import torch
import torch.nn as nn
import torch.nn.functional as F

import random


PAD_token = 0
SOS_token = 1
EOS_token = 2


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidn_dim, device, dropout=0.5):
        super().__init__()

        self.device = device

        self.input_dim = input_dim
        self.hidn_dim = hidn_dim
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)  
        self.lstm = nn.LSTM(emb_dim, hidn_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs, prev_hidden, prev_cell):
        """
        inputs: (batch_size, one_hot_encode, hidn_dim)
        prev_hidden: (batch_size, hidn_dim, hidn_dim)
        embedded: (batch_size, seq_len, hidn_size)
        output: (batch_size, seq_len, hidn_size)
        hidden: (batch_size, seq_len, hidn_size)
        cell: (batch_size, seq_len, hidn_size)
        """
        inputs_gpu = inputs.to(self.device)
        embedded = self.embedding(inputs_gpu)
        output, (hidden, cell) = self.lstm(embedded, (prev_hidden, prev_cell))
        output = self.dropout(output)
        
        return output, hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidn_dim, device):
        super().__init__()
        
        self.device = device
        
        self.output_dim = output_dim
        self.hidn_dim = hidn_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidn_dim, batch_first=True)
        self.out = nn.Linear(hidn_dim, output_dim)
    
    def forward(self, inputs, prev_hidden, prev_cell):
        """
        * use batch_first = True
        inputs: (batch_size, 1)
        embedded: (batch_size, 1, hidn_size)
        output: (batch_size, 1, hidn_size)
        hidden: (batch_size, 1, hidn_size)
        cell: (batch_size, 1, hidn_size)
        """
        inputs = inputs.unsqueeze(1)
        inputs_gpu = inputs.to(self.device)
        embedded = self.embedding(inputs_gpu)
        output = F.relu(embedded)
        output, (hidden, cell) = self.lstm(output, (prev_hidden, prev_cell))

        output = self.out(output)

        output = output.squeeze(1)

        return output, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, inputs, targets, teacher_forcing_ratio = 0.5):   
        batch_size = targets.size(0)
        input_len = inputs.size(1)
        target_len = targets.size(1)
        enc_hidn_dim = self.encoder.hidn_dim
        tar_vocab_size = self.decoder.output_dim

        enc_hidn = torch.zeros(1, batch_size, enc_hidn_dim, device=self.device)
        enc_cell = torch.zeros(1, batch_size, enc_hidn_dim, device=self.device)
        dec_outputs = torch.zeros(batch_size, target_len, tar_vocab_size, device=self.device)
        predictions = torch.zeros(batch_size, target_len, device=self.device)
        
        # Encode
        enc_outputs, enc_hidn, enc_cell = self.encoder.forward(inputs, enc_hidn, enc_cell)
        # Decode
        dec_input = targets[:, 0] 
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        
        predictions[:, 0] = dec_input
        
        for i in range(1, target_len):
            dec_output, dec_hidn, dec_cell = self.decoder.forward(dec_input, enc_hidn, enc_cell)

            dec_outputs[:, i] = dec_output
            target = targets[:, i]

            if use_teacher_forcing:
                dec_input = target
            else:
                dec_input = dec_output.argmax(1)            
            predictions[:, i] = dec_input
            
        return dec_outputs, predictions
    
    
class Regressor(nn.Module):
    def __init__(self, input_dim, hidn_dim, device, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.device = device
        self.dense_hidn = nn.Linear(input_dim, hidn_dim)
        self.linear = nn.Linear(hidn_dim, 16)
        self.dense_out = nn.Linear(16, 1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, enc_output):
        """ 
        inputs: sequence latent representation
            - (batch_size, seq_len, hidn_size)
        targets: RFU, CT
            - (batch_size, 1)
        """

        # encoder output -> regression
        pool = nn.MaxPool1d(enc_output.shape[1])
        output = pool(enc_output.permute(0,2,1)).permute(0,2,1)
        output = self.flatten(output)
        output = self.dropout(self.dense_hidn(output))
        output = self.dropout(self.linear(output))
        output = self.dense_out(output)
        
        return output
