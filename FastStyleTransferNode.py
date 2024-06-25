""" 
This node is a simple conversion of this repository into ComfyUI ecosystem:
https://github.com/rrmina/fast-neural-style-pytorch.git

Some of the code is written by ChatGPT4-o
"""


import torch
import torch.nn as nn
import time
import os
import folder_paths

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm="instance"):
        super(ConvLayer, self).__init__()
        # Padding Layers
        padding_size = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding_size)

        # Convolution Layer
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        # Normalization Layers
        self.norm_type = norm
        if (norm=="instance"):
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        elif (norm=="batch"):
            self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv_layer(x)
        if (self.norm_type=="None"):
            out = x
        else:
            out = self.norm_layer(x)
        return out


class ResidualLayer(nn.Module):
    """
    Deep Residual Learning for Image Recognition

    https://arxiv.org/abs/1512.03385
    """
    def __init__(self, channels=128, kernel_size=3):
        super(ResidualLayer, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size, stride=1)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size, stride=1)

    def forward(self, x):
        identity = x                     # preserve residual
        out = self.relu(self.conv1(x))   # 1st conv layer + activation
        out = self.conv2(out)            # 2nd conv layer
        out = out + identity             # add residual
        return out


class DeconvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding, norm="instance"):
        super(DeconvLayer, self).__init__()

        # Transposed Convolution 
        padding_size = kernel_size // 2
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding_size, output_padding)

        # Normalization Layers
        self.norm_type = norm
        if (norm=="instance"):
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        elif (norm=="batch"):
            self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.conv_transpose(x)
        if (self.norm_type=="None"):
            out = x
        else:
            out = self.norm_layer(x)
        return out


class TransformerNetwork(nn.Module):
    """Feedforward Transformation Network without Tanh
    reference: https://arxiv.org/abs/1603.08155 
    exact architecture: https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf
    """
    def __init__(self):
        super(TransformerNetwork, self).__init__()
        self.ConvBlock = nn.Sequential(
            ConvLayer(3, 32, 9, 1),
            nn.ReLU(),
            ConvLayer(32, 64, 3, 2),
            nn.ReLU(),
            ConvLayer(64, 128, 3, 2),
            nn.ReLU()
        )
        self.ResidualBlock = nn.Sequential(
            ResidualLayer(128, 3), 
            ResidualLayer(128, 3), 
            ResidualLayer(128, 3), 
            ResidualLayer(128, 3), 
            ResidualLayer(128, 3)
        )
        self.DeconvBlock = nn.Sequential(
            DeconvLayer(128, 64, 3, 2, 1),
            nn.ReLU(),
            DeconvLayer(64, 32, 3, 2, 1),
            nn.ReLU(),
            ConvLayer(32, 3, 9, 1, norm="None")
        )

    def forward(self, x):
        x = self.ConvBlock(x)
        x = self.ResidualBlock(x)
        out = self.DeconvBlock(x)
        return out



class FastStyleTransfer:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "content_img": ("IMAGE",),
                "model": ([file for file in os.listdir(os.path.join(folder_paths.base_path, "custom_nodes\\ComfyUI-Fast-Style-Transfer\\models\\")) if file.endswith('.pth')], ),
                },
            }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "styleTransfer"

    CATEGORY = "Style Transfer"


    def encode_tensor(self, tensor):
        tensor = tensor.permute(0, 3, 1, 2).contiguous()  # Convert to [batch_size, channels, height, width]
        return tensor * 255

    def decode_tensor(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1).contiguous()  # Convert to [batch_size, height, width, channels]
        return tensor / 255
    
    def styleTransfer(self, content_img, model):
        # Device
        device = ("cuda" if torch.cuda.is_available() else "cpu")

        # Load Transformer Network
        net = TransformerNetwork().to(device)
        model_path = os.path.join(folder_paths.base_path, "custom_nodes\\ComfyUI-Fast-Style-Transfer\\models\\") + model
        net.load_state_dict(torch.load(model_path, map_location=device))
        net = net.to(device)

        with torch.no_grad():
            torch.cuda.empty_cache()
            starttime = time.time()
            content_tensor = self.encode_tensor(content_img)
            generated_tensor = net(content_tensor.to(device))
            print("Transfer Time: {}".format(time.time() - starttime))
            image = self.decode_tensor(generated_tensor)
        return (image,)
    

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "FastStyleTransfer": FastStyleTransfer
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "FastStyleTransfer": " Fast Style Transfer"
}
