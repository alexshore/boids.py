import numpy as np

import torch
import torch.nn as nn

import time


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layer1 = nn.Linear(in_features=8, out_features=16, bias=False)
        # self.layer1 = nn.Linear(in_features=6, out_features=16, bias=False)
        self.layer2 = nn.Linear(in_features=16, out_features=8, bias=False)
        self.layer3 = nn.Linear(in_features=8, out_features=4, bias=False)
        self.layer4 = nn.Linear(in_features=4, out_features=2, bias=False)

    @torch.no_grad()
    def forward(self, x) -> torch.Tensor:
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        x = torch.relu(x)
        x = self.layer4(x)
        x = torch.sigmoid(x)
        return x


def convert_weights_to_torch(weights) -> Network:
    network = Network()
    state_dict = {
        "layer1.weight": torch.tensor(np.reshape(weights[:128], (16, 8))),
        "layer2.weight": torch.tensor(np.reshape(weights[128:256], (8, 16))),
        "layer3.weight": torch.tensor(np.reshape(weights[256:288], (4, 8))),
        "layer4.weight": torch.tensor(np.reshape(weights[288:], (2, 4))),
    }
    network.load_state_dict(state_dict)
    return network


def main():
    avg_time = 0
    for _ in range(100):
        network = convert_weights_to_torch(np.random.normal(0.0, 0.5, 296))
        start = time.time()
        output = network.forward(torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.float, requires_grad=False))
        print(output)
        avg_time += (time.time() - start) / 100
    print(avg_time)


if __name__ == "__main__":
    main()
