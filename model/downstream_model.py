from base import BaseModel
import torch.nn as nn


class ClassificationModel(BaseModel):
    def __init__(self, input_dim: int = 512, hidden_dim: int = 512, output_dim: int = 1):
        super(ClassificationModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class RegressionModel(nn.Module):
    def __init__(self, input_dim: int = 512, hidden_dim: int = 512, output_dim: int = 1):
        super(RegressionModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(self, x):
        x = self.layers(x)
        return x
