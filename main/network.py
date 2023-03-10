import torch.nn as nn


class Network(nn.Module):
    def __init__(self) -> None:
        super(Network, self).__init__()

        """
        in features:

        distance to nearest obstacle in 5 forward facing directions.
        distance to nearest 3 boids
        current velocity
        

        """

        self.layers = nn.Sequential(
            nn.Linear(in_features=1, out_features=1),
        )
