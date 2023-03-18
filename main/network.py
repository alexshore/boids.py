import numpy as np

import torch
import torch.nn as nn

import time


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layer1 = nn.Linear(in_features=14, out_features=10, bias=False)
        self.layer2 = nn.Linear(in_features=10, out_features=5, bias=False)
        self.layer3 = nn.Linear(in_features=5, out_features=1, bias=False)

    # @torch.no_grad()
    def forward(self, x) -> torch.Tensor:
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        x = torch.tanh(x)
        return x


def convert_weights_to_torch(weights) -> nn.Module:
    network = Network()
    state_dict = {
        "layer1.weight": torch.tensor(np.reshape(weights[:140], (10, 14))),
        "layer2.weight": torch.tensor(np.reshape(weights[140:190], (5, 10))),
        "layer3.weight": torch.tensor(np.reshape(weights[190:], (1, 5))),
    }
    network.load_state_dict(state_dict)
    return network


def main():
    network = convert_weights_to_torch(np.random.normal(0.0, 0.5, 195))
    start = time.time()
    output = network.forward(torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.float, requires_grad=False))
    print(output.item())
    print(time.time() - start)


if __name__ == "__main__":
    main()
