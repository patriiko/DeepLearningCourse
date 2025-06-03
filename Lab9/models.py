import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class MojModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_units=256, drop_prob=0.2, num_layers=2):
        super(MojModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.num_layers = num_layers

        # Embedding layer: maps word indices to dense vectors
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # GRU layer
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_units, 
                         num_layers=num_layers, dropout=drop_prob, batch_first=True)
        # Fully connected layer to map GRU output to vocabulary size
        self.fc = nn.Linear(hidden_units, vocab_size)
        # Softmax for output probabilities
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input):
        # Input shape: (batch_size, maxlen)
        emb = self.embeddings(input)  # Shape: (batch_size, maxlen, embedding_dim)
        
        # GRU layer
        self.gru.flatten_parameters()  # For multi-GPU compatibility
        gru_out, _ = self.gru(emb)  # Shape: (batch_size, maxlen, hidden_units)
        
        # Take the output of the last time step
        out = gru_out[:, -1, :]  # Shape: (batch_size, hidden_units)
        
        # Fully connected layer
        out = self.fc(out)  # Shape: (batch_size, vocab_size)
        
        # Apply softmax
        out = self.softmax(out)  # Shape: (batch_size, vocab_size)
        return out