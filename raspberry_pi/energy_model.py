import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

history_size = 60
n_predictions = 60

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTMPredictor(nn.Module):
    def __init__(self, n_hidden=51):
        super(LSTMPredictor, self).__init__()
        self.n_hidden = n_hidden
        # lstm1, lstm2, linear
        self.lstm1 = nn.LSTMCell(1, self.n_hidden)  # recieve only one value (power consumption)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 1)  # output only one value

    def forward(self, x):  # forward pass
        outputs = []
        n_samples = x.size(0)

        h_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32, device=device)  # inicial hidden state
        c_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32, device=device)  # inicial cell state
        h_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32, device=device)  # second hidden state
        c_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32, device=device)  # second cell state

        for input_t in x.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)
        return outputs

    def generate_prediction(self, day):  # returns the current day chosen and the predictions for the next day
        y = self.read_dataset()

        y = np.mean(y[day - 2 if day > 2 else 0:day + 1], axis=0)  # data preparation
        y = torch.from_numpy(np.array([y])).to(device)
        out = self.forward(y).detach().cpu().numpy().flatten().clip(0) * 2  # generating predictions
        y = y.detach().cpu().numpy().flatten()

        return y, out

    @staticmethod
    def read_dataset():  # returns the dataset
        db = np.zeros((6, 60), dtype=np.float32)

        with open('./energy_dataset.prjt', 'r') as file:
            content = file.read().splitlines()
            for idx, line in enumerate(content):
                db[idx] = np.float_(line.split(';'))

        return db


def energy_pred():  # returns the loaded model
    model = LSTMPredictor(n_hidden=128)
    model.to(device)
    model.load_state_dict(torch.load('./energy_params.nn'))  # load the parameters into the model
    model.eval()
    return model


if __name__ == '__main__':
    model = energy_pred()

    for i in range(8):
        y, out = model.generate_prediction(i)

        plt.figure(figsize=(10, 8))
        plt.title(f'Graph {i + 1}')
        plt.xlabel('Medidas')
        plt.ylabel('KW')

        p = len(y)

        plt.plot(np.arange(p), y, 'g', linewidth=1, label=f'Dia {i+1}')
        plt.plot(np.arange(p, p + len(y)), out, 'r', linewidth=2, label=f'Prev. Dia {i+2}')

        plt.legend()
        plt.show()
        plt.close()
