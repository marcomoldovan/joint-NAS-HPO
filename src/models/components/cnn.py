import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
    
    
class FeedForwardBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        use_activation: bool = True,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU() if use_activation else nn.Identity(),
            nn.Dropout(dropout_rate),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
    
    
class CNN(nn.Module):
    def __init__(
        self,
        conv_channels: list = [1, 16, 32, 64],
        kernel_size: int = 5,
        stride: int = 1,
        use_batch_norm: bool = True,
        fc_channels: list = [128, 64, 10],
    ) -> None:
        """_summary_

        Args:
            conv_channels (list, optional): _description_. Defaults to [1, 16, 32, 64].
            kernel_size (int, optional): _description_. Defaults to 5.
            stride (int, optional): _description_. Defaults to 1.
            use_batch_norm (bool, optional): _description_. Defaults to True.
            fc_channels (list, optional): List containing the output dimensions of all fully connected layers. Has to contain at list the given default as those are the finals output classes. Number of fully connected layers is given by length of this list. Defaults to [10].
        """
        super().__init__()
        
        num_conv_layers = len(conv_channels) - 1
        conv_blocks = []
        for i in range(num_conv_layers):
            padding = (kernel_size - 1) // 2
            block = CNNBlock(conv_channels[i], conv_channels[i+1], kernel_size, stride, padding, use_batch_norm)
            conv_blocks.append(block)
        self.conv = nn.Sequential(*conv_blocks)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        num_fc_layers = len(fc_channels)
        fc_blocks = []
        for i in range(num_fc_layers):
            # only layer
            if i == 0 and num_fc_layers == 1:
                block = FeedForwardBlock(conv_channels[-1], fc_channels[i], use_activation=False)
            # first layer with others to follow
            elif i == 0 and num_fc_layers > 1:
                block = FeedForwardBlock(conv_channels[-1], fc_channels[i])
            # last layer with others before it
            elif i == num_fc_layers - 1:
                block = FeedForwardBlock(fc_channels[i-1], fc_channels[i], use_activation=False)
            # all other layers
            else:
                block = FeedForwardBlock(fc_channels[i-1], fc_channels[i])
            fc_blocks.append(block)
        self.fc = nn.Sequential(*fc_blocks)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1) # or x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        

if __name__ == "__main__":
    _ = CNN()

