import torch.nn as nn
class TemporalLSTM(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, num_layers=2, output_dim=2):
        super(TemporalLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return self.softmax(out)
