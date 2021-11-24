import numpy as np
import torch
import torch.nn as nn
import torch.optim as optiom
import matplotlib.pyplot as plt

learning_rate = 0.8
n_steps = 20
n_predictions = 144
print_every = 1

N = 250  # number of sine waves generated
L = 144  # number of points per wave
T = 240  # period-like

x = np.empty((N, L), np.float32)
x[:] = np.array(range(L)) * 10 - T - 100
y = np.sin(x / T).astype(np.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # if available will run on the GPU (much faster for matrix processing)


class predictor(nn.Module):
    def __init__(self, n_hidden=51):
        super(predictor, self).__init__()
        self.n_hidden = n_hidden
        # lstm1, lstm2, linear
        self.lstm1 = nn.LSTMCell(1, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 1)

    def forward(self, x, future=0):
        outputs = []
        n_samples = x.size(0)

        h_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)  # hidden state
        c_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)  # cell state
        h_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)  # hidden state
        c_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)  # cell state

        h_t = h_t.to(device)
        c_t = c_t.to(device)
        h_t2 = h_t2.to(device)
        c_t2 = c_t2.to(device)

        for input_t in x.split(1, dim=1):  # calculate one-by-one
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)

        for i in range(future):  # predict next 1000 points
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)
        return outputs


if __name__ == '__main__':
    # y = 100, 1000
    train_input = torch.from_numpy(y[3:, :-1])  # 97, 999
    train_target = torch.from_numpy(y[3:, 1:])  # 97, 999

    test_input = torch.from_numpy(y[:3, :-1])  # 3, 999
    test_target = torch.from_numpy(y[:3, 1:])  # 3, 999

    train_input = train_input.to(device)  # uses the GPU if available
    train_target = train_target.to(device)

    test_input = test_input.to(device)
    test_target = test_target.to(device)

    model = predictor(n_hidden=64)

    model.to(device)  # uses the GPU if available

    criterion = nn.MSELoss()
    optimizer = optiom.LBFGS(model.parameters(), lr=learning_rate, history_size=144)

    for i in range(n_steps):
        print(f'Step: {i}')


        def closure():
            optimizer.zero_grad()
            out = model(train_input)
            loss = criterion(out, train_target)
            print(f'Loss: {loss.item()}')
            loss.backward()
            return loss


        optimizer.step(closure)

        with torch.no_grad():
            pred = model(test_input, future=n_predictions)
            loss = criterion(pred[:, :-n_predictions], test_target)
            print(f'Test loss: {loss.item()}')
            y = pred.detach().cpu().numpy()

        if i % print_every == 0:
            plt.figure(figsize=(10, 8))
            plt.title(f'Step {i + 1}')
            plt.xlabel('X')
            plt.ylabel('Y')
            n = train_input.shape[1]  # 999


            def draw(y_i, color):
                plt.plot(np.arange(n), y_i[:n], color, linewidth=2)
                plt.plot(np.arange(n, n + n_predictions), y_i[n:], color + ":", linewidth=2)  # predictions


            # draw(y[0], 'r')  # only show 1 for better visualization
            draw(y[1], 'b')
            # draw(y[2], 'g')

            plt.show()  # shows graph
            # plt.savefig(f'predict{i + 1}.pdf')  # saves graph as pdf
            plt.close()

    torch.save(model.state_dict(), './params.nn')
