import torch.nn as nn
from .helper import ResidualBlock, NonLocalBlock, DownSampleBlock, GroupNorm, Swish


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        channels = [128, 128, 128, 256, 256, 512]
        attn_resolutions = [16]
        num_res_blocks = 2
        layers = [nn.Conv2d(args.image_channels, channels[0], 3, 1, 1)]
        resolution = 256
        for i in range(len(channels)-1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            if i != len(channels) - 2:
                layers.append(DownSampleBlock(channels[i+1]))
                resolution //= 2
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(NonLocalBlock(channels[-1]))
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(GroupNorm(channels[-1]))
        layers.append(Swish())
        layers.append(nn.Conv2d(channels[-1], args.latent_dim, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
'''
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.channels = [128, 128, 128, 256, 256, 512]
        self.attn_resolutions = [16]
        self.num_res_blocks = 2
        self.resolution = 256

        # Define the initial convolution layers for 1-channel and 4-channel inputs
        self.input_conv_1 = nn.Conv2d(1, self.channels[0], 3, 1, 1)  # For single-channel images
        self.input_conv_4 = nn.Conv2d(args.image_channels, self.channels[0], 3, 1, 1)  # For four-channel masks

        # Define the rest of the layers
        self.layers = self._build_main_layers()

        # Final layers after main processing
        self.final_layers = nn.Sequential(
            ResidualBlock(self.channels[-1], self.channels[-1]),
            NonLocalBlock(self.channels[-1]),
            ResidualBlock(self.channels[-1], self.channels[-1]),
            GroupNorm(self.channels[-1]),
            Swish(),
            nn.Conv2d(self.channels[-1], args.latent_dim, 3, 1, 1)
        )

    def _build_main_layers(self):
        layers = []
        resolution = self.resolution

        for i in range(len(self.channels) - 1):
            in_channels = self.channels[i]
            out_channels = self.channels[i + 1]
            for j in range(self.num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in self.attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            if i != len(self.channels) - 2:
                layers.append(DownSampleBlock(out_channels))
                resolution //= 2

        return nn.Sequential(*layers)

    def forward(self, x):
        # Determine which input convolution to use based on the number of channels
        if x.shape[1] == 1:
            x = self.input_conv_1(x)
        elif x.shape[1] == 4:
            x = self.input_conv_4(x)
        else:
            raise ValueError("Unexpected number of input channels: expected 1 or 4.")

        # Pass through the main layers
        x = self.layers(x)
        
        # Pass through the final layers
        x = self.final_layers(x)
        return x
'''