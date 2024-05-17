import torch
import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, char_to_index, index_to_char, input_size, hidden_size, output_size, num_layers):
        super(CharRNN, self).__init__()
        self.char_to_index = char_to_index
        self.index_to_char = index_to_char
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)


class CharLSTM(nn.Module):
    def __init__(self, char_to_index, index_to_char, input_size, hidden_size, output_size, num_layers):
        super(CharLSTM, self).__init__()
        self.char_to_index = char_to_index
        self.index_to_char = index_to_char
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))