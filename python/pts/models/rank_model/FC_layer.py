import torch.nn as nn


class FC_layer(nn.Module):
    """MLP module for classification"""

    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(FC_layer, self).__init__()
        self.layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        """MLP classifier
        :param x: (BS,  dim)
        :return x: (BS, class_num)
        """
        return self.layers(x)