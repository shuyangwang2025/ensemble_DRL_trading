import torch
from torch import nn
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)


class IndependentNormalModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.n_past = hparams.n_past
        self.num_features = hparams.n_features
        self.num_actions = hparams.n_actions
        self.num_coins = len(hparams.coin_list)
        lstm_n_units = hparams.lstm_n_units
        lstm_n_layers = hparams.lstm_n_layers

        self.fc1 = nn.Linear(self.num_features, hparams.fc_units)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.lstm = nn.LSTM(hparams.fc_units, lstm_n_units, batch_first=True, num_layers=lstm_n_layers)
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        self.bn1 = nn.BatchNorm1d(lstm_n_units)
        self.loc_layer = nn.Linear(lstm_n_units, self.num_coins)
        nn.init.xavier_uniform_(self.loc_layer.weight)
        self.scale_layer = nn.Linear(lstm_n_units, self.num_coins)
        nn.init.xavier_uniform_(self.scale_layer.weight)

        self.value = nn.Linear(lstm_n_units, 1)
        nn.init.xavier_uniform_(self.value.weight)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.const_ep = torch.tensor(0.0001, device=device)

    def forward(self, obs: torch.Tensor):
        obs = obs.reshape((-1, self.n_past, self.num_features)) # Compatible with batch first
        out = self.fc1(obs)
        out = F.sigmoid(out)
        out, hidden1 = self.lstm(out)
        out = out[:,-1,:]
        out = self.bn1(out)

        loc_out = self.loc_layer(out)
        loc_out = F.tanh(loc_out)

        scale_out = self.scale_layer(out)
        scale_out = F.sigmoid(scale_out) + self.const_ep

        value = self.value(out).reshape(-1)

        return loc_out, scale_out, value