import torch
import torch.nn as nn

class LSTMRegressionModel(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, embedding_pretrained=None) -> None:
        super(LSTMRegressionModel, self).__init__()
        if embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=vocab_size-2)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        # [batch_size, seq_len] -> [batch_size, seq_len, embedding_size]
        x = self.embedding(inputs)
        # [batch_size, seq_len, embed_dim] -> [num_layers, batch_size, hidden_size]
        _, (x, _) = self.lstm(x)
        x = x[-1, :, :]
        # [batch_size, hidden_size] -> [batch_size, 1]
        x = self.linear(x)
        out = self.sigmoid(x)
        return out
