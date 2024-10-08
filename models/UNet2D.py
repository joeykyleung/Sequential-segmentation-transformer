# Adapted from "Comparing 3D, 2.5D, and 2D Approaches to Brain Image Auto-Segmentation"

import torch
from torch import nn
from torch.nn.functional import pad


class Config:
    ''' Configurations for the 2D UNet '''
    name = 'UNET2D'
    slice_size = (256, 256, 21)
    
    batch_size = 12
    loss_type = 'diceCE'
    optimizer = 'adam'
    learning_rate = 1e-4
    
    validation_freq = 100


class UNet2D(nn.Module):
    ''' Architecture for a 2D UNet '''
    
    def __init__(self, Ci=1, Co=1, xpad=True):
        """
        Inputs:
            - Ci: number of input channels into the UNet
            - Co: number of output channels out of the UNet
            - xpad: if the input size is not 2^n in all dimensions, set xpad to True.
        """
        super().__init__()

        # Downsampling limb of UNet:
        self.left1 = Doubleconv(Ci, 64)
        self.left2 = DownDoubleconv(64, 128)
        self.left3 = DownDoubleconv(128, 256)
        self.left4 = DownDoubleconv(256, 512)

        # Bottleneck of UNet:
        self.bottom = DownDoubleconv(512, 1024)

        # Upsampling limb of UNet:
        # the units are numbered in reverse to match the corresponding downsampling units
        self.right4 = UpConcatDoubleconv(1024, 512, xpad)
        self.right3 = UpConcatDoubleconv(512, 256, xpad)
        self.right2 = UpConcatDoubleconv(256, 128, xpad)
        self.right1 = UpConcatDoubleconv(128, 64, xpad)

        self.out = Outconv(64, Co)


    def forward(self, x):
        """
        Input:
            - x: UNet input; type: torch tensor; dimensions: [B, Ci, H, W]
        Output:
            - UNet output; type: torch tensor; dimensions: output[B, Co, H, W]
        Dimensions explained:
            - 'i' and 'o' subscripts repsectively mean inputs and outputs.
            - C: channels
            - H: height
            - W: width
        """
        # Downsampling limb of UNet:
        x1 = self.left1(x)
        x2 = self.left2(x1)
        x3 = self.left3(x2)
        x4 = self.left4(x3)

        # Bottleneck of UNet:
        x = self.bottom(x4)

        # Upsampling limb of UNet:
        x = self.right4(x4, x)
        x = self.right3(x3, x)
        x = self.right2(x2, x)
        x = self.right1(x1, x)

        return self.out(x)


class Doubleconv(nn.Module):
    """
    DoubleConvolution units in the UNet
    """

    def __init__(self, Ci, Co):
        """
        Inputs:
            - Ci: number of input channels into the DoubleConvolution unit
            - Co: number of output channels out of the DoubleConvolution unit
        """
        super().__init__()
        self.doubleconv = nn.Sequential(
            nn.Conv2d(Ci, Co, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(Co),
            nn.ReLU(inplace=True),
            nn.Conv2d(Co, Co, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(Co),
            nn.ReLU(inplace=True))

    def forward(self, x):
        """
        Input:
        - x: torch tensor; dimensions: [B, Ci, Hi, Wi]
        Output:
            - return: x --> conv2d --> batch_norm --> ReLU --> conv2d --> batch_norm --> ReLU --> output
                dimensions: [B, Co, Ho, Wo]
        """
        return self.doubleconv(x)


class DownDoubleconv(nn.Module):
    """
    Units in the left side of the UNet:
    Down-sample using MaxPool2d --> then DoubleConvolution
    """

    def __init__(self, Ci, Co):
        """
        Inputs:
            - Ci: number of input channels into the DownDoubleconv unit
            - Co: number of output channels out of the DownDoubleconv unit
        """
        super().__init__()
        self.maxpool_doubleconv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Doubleconv(Ci, Co))


    def forward(self, x):
        """
        Input:
            - x: torch tensor; dimensions: x[batch, channels, H, W]
        Output:
            - return: x --> maxpool2d --> DoubleConv Unit --> output
        """
        return self.maxpool_doubleconv(x)


class UpConcatDoubleconv(nn.Module):
    """
    Units in the right side of the UNet:
    Up-scale using ConvTranspose2d --> Concatenate the bottom and horizontal channels --> DoubleConvolution
    """

    def __init__(self, Ci, Co, xpad=True, up_mode='transposed'):
        """
        Inputs:
            - Ci: number of input channels into the Up unit
            - Co: number of output channels out of the Up unit
            - xpad: set this to False only if the input H/W dimensions are all powers of two. Otherwise set
                            this to True.
            - up_mode: default is 'transposed'. Set this to 'trilinear' if you want trilinear interpolation
                            instead (but interpolation would make the network slow).
        """
        super().__init__()
        self.xpad = xpad
        self.up_mode = up_mode

        if self.up_mode == 'transposed':
            self.up = nn.ConvTranspose2d(Ci, Co, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode=self.up_mode, align_corners=True)

        self.doubleconv = Doubleconv(Ci, Co)


    def forward(self, x1, x2):
        """
        Inputs:
            - x1: skip-connection from the downsampling limb of U-Net; dimensions: [B, C1, H1, W1]
            - x2: from the lower-level upsampling limb of U-Net; dimensions: [B, C2, H2, W2]
        Output:
            - return: up-scale x2 --> concatenate(x1, x2) --> DoubleConv Unit --> output
        """
        x2 = self.up(x2)

        if self.xpad:
            # If H2 or W2 are smaller than H1 or W1, pad x2 so that its size matches x1.
            B1, C1, H1, W1 = x1.shape
            B2, C2, H2, W2 = x2.shape
            assert B1 == B2
            diffH, diffW = H1 - H2, W1 - W2
            x2 = pad(x2, [diffW // 2, diffW - diffW // 2,
                        diffH // 2, diffH - diffH // 2])

        # Concatenate x1 and x2:
        x = torch.cat([x1, x2], dim=1)

        # Return double convolution of the concatenated tensor:
        return self.doubleconv(x)

# ........................................................................................................

class Outconv(nn.Module):
    """
    Output unit in the UNet
    """

    def __init__(self, Ci, Co):
        """
        Inputs:
            - Ci: number of input channels into the final output unit
            - Co: number of output channels out of the entire UNet
        """
        super().__init__()
        self.conv_sigmoid = nn.Sequential(
            nn.Conv2d(Ci, Co, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x):
        return self.conv_sigmoid(x)


if __name__ == '__main__':

    from torchsummary import summary

    size = (256, 256)
    
    x = torch.rand(1, 1, *size)  # batch of 1 MRI volume: 1 channel, 212 x 212 voxels
    model = UNet2D()
    preds = model(x)
    print(f'Input shape: {x.shape} \n'
          f'Output shape: {preds.shape}')
    print(f'Input and output are the same shape? {preds.shape == x.shape}')

    summary(model, (1, *size)) # following "An Exploration of 2D and 3D Deep Learning Techniques..."
    # for summary, the second argument is the shape of each input data (not the batch).
