""" 
These nodes are a simple conversion of these repositories into ComfyUI ecosystem:
https://github.com/rrmina/fast-neural-style-pytorch.git
https://github.com/gordicaleksa/pytorch-neural-style-transfer.git

"""

import torch
import torch.nn as nn
import time
import os
import folder_paths
import subprocess as sp
import sys

# ML classes
class ConvolutionalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm="instance"):
        super(ConvolutionalLayer, self).__init__()
        # Padding Layers
        self.padding_size = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(self.padding_size)

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
        if self.norm_type == "None":
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
        self.conv1 = ConvolutionalLayer(channels, channels, kernel_size, stride=1)
        self.relu = nn.ReLU()
        self.conv2 = ConvolutionalLayer(channels, channels, kernel_size, stride=1)

    def forward(self, x):
        identity = x                     # preserve residual
        out = self.relu(self.conv1(x))   # 1st conv layer + activation
        out = self.conv2(out)            # 2nd conv layer
        out = out + identity             # add residual
        return out


class DeconvolutionalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding, norm="instance"):
        super(DeconvolutionalLayer, self).__init__()

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
            ConvolutionalLayer(3, 32, 9, 1),
            nn.ReLU(),
            ConvolutionalLayer(32, 64, 3, 2),
            nn.ReLU(),
            ConvolutionalLayer(64, 128, 3, 2),
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
            DeconvolutionalLayer(128, 64, 3, 2, 1),
            nn.ReLU(),
            DeconvolutionalLayer(64, 32, 3, 2, 1),
            nn.ReLU(),
            ConvolutionalLayer(32, 3, 9, 1, norm="None")
        )

    def forward(self, x):
        x = self.ConvBlock(x)
        x = self.ResidualBlock(x)
        out = self.DeconvBlock(x)
        return out


# Node classes
class TrainFastStyleTransfer:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "style_img": ("IMAGE",),
                "seed": ("INT", {"default": 30, "min": 0, "max": 999999, "step": 1,}),
                "content_weight": ("INT", {"default": 14, "min": 1, "max": 128, "step": 1,}),
                "style_weight": ("INT", {"default": 50, "min": 1, "max": 128, "step": 1,}),
                "tv_weight": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.0000001}),
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 32, "step": 1,}),
                "train_img_size": ("INT", {"default": 256, "min": 128, "max": 2048, "step": 1,}),
                "learning_rate": ("FLOAT", {"default": 0.001, "min": 0.0001, "max": 100.0, "step": 0.0001}),
                "num_epochs": ("INT", {"default": 1, "min": 1, "max": 20, "step": 1,}),
                "save_model_every": ("INT", {"default": 500, "min": 10, "max": 10000, "step": 10,}),
                "from_pretrained": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1,}),
                "model": ([file for file in os.listdir(os.path.join(folder_paths.base_path, "custom_nodes/ComfyUI-Fast-Style-Transfer/models/")) if file.endswith('.pth')], ),
                },
            }

    RETURN_TYPES = ()
    OUTPUT_NODE = True

    FUNCTION = "train"

    CATEGORY = "Style Transfer"

    def encode_tensor(self, tensor):
        tensor = tensor.permute(0, 3, 1, 2).contiguous()  # Convert to [batch_size, channels, height, width]
        return tensor[:, [2, 1, 0], :, :] * 255


    def train(self, style_img, seed, batch_size, train_img_size, learning_rate, num_epochs, content_weight, style_weight, tv_weight, save_model_every, from_pretrained, model):
        temp_save_style_img = os.path.join(folder_paths.base_path, "custom_nodes/ComfyUI-Fast-Style-Transfer/temp/") + "temp_save_content_img.pt"
        save_model_path = os.path.join(folder_paths.base_path, "custom_nodes/ComfyUI-Fast-Style-Transfer/models/")
        dataset_path = os.path.join(folder_paths.base_path, "custom_nodes/ComfyUI-Fast-Style-Transfer/dataset/")
        vgg_path = os.path.join(folder_paths.base_path, "custom_nodes/ComfyUI-Fast-Style-Transfer/vgg/vgg16-00b39a1b.pth")
        save_image_path = os.path.join(folder_paths.base_path, "custom_nodes/ComfyUI-Fast-Style-Transfer/output/")
        train_path = os.path.join(folder_paths.base_path, "custom_nodes/ComfyUI-Fast-Style-Transfer/train.py")
        


        command = [
            sys.executable, train_path,
            '--train_image_size', str(train_img_size),
            '--dataset_path', dataset_path,
            '--vgg_path', vgg_path,
            '--num_epochs', str(num_epochs),
            '--temp_save_style_img', temp_save_style_img,
            '--batch_size', str(batch_size),
            '--content_weight', str(content_weight),
            '--style_weight', str(style_weight),
            '--tv_weight', str(tv_weight),
            '--adam_lr', str(learning_rate),
            '--save_model_path', save_model_path,
            '--save_image_path', save_image_path,
            '--save_model_every', str(save_model_every),
            '--seed', str(seed),
            '--pretrained_model'
        ]

        if from_pretrained:
            command.append(model)
        else:
            command.append('none')
        
        
        torch.save(self.encode_tensor(style_img), temp_save_style_img)
        
        sp.run(command)
        return ()


class FastStyleTransfer:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "content_img": ("IMAGE",),
                "model": ([file for file in os.listdir(os.path.join(folder_paths.base_path, "custom_nodes/ComfyUI-Fast-Style-Transfer/models/")) if file.endswith('.pth')], ),
                },
            }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "styleTransfer"

    CATEGORY = "Style Transfer"


    def encode_tensor(self, tensor):
        tensor = tensor.permute(0, 3, 1, 2).contiguous()  # Convert to [batch_size, channels, height, width]
        return tensor[:, [2, 1, 0], :, :] * 255

    def decode_tensor(self, tensor):
        tensor = tensor[:, [2, 1, 0], :, :]
        tensor = tensor.permute(0, 2, 3, 1).contiguous()  # Convert to [batch_size, height, width, channels]
        return tensor / 255
    
    def styleTransfer(self, content_img, model):
        # Device
        device = ("cuda" if torch.cuda.is_available() else "cpu")

        # Load Transformer Network
        net = TransformerNetwork().to(device)
        model_path = os.path.join(folder_paths.base_path, "custom_nodes/ComfyUI-Fast-Style-Transfer/models/") + model
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
    

class NeuralStyleTransfer:



    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "content_img": ("IMAGE",),
                "style_img": ("IMAGE",),
                "content_weight": ("FLOAT", {"default": 1e5, "min": 1e3, "max": 1e6, "step": 1e3}),
                "style_weight": ("FLOAT", {"default": 3e4, "min": 1e1, "max": 1e5, "step": 1e1}),
                "tv_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1e1, "step": 0.1}),
                "num_steps": ("INT", {"default": 100, "min": 10, "max": 10000, "step": 10}),
                "learning_rate": ("FLOAT", {"default": 1.0, "min": 1e-4, "max": 1e3, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "neural_style_transfer"
    CATEGORY = "Style Transfer"
    
    def encode_tensor(self, tensor):
        tensor = tensor.permute(0, 3, 1, 2).contiguous()  # Convert to [batch_size, channels, height, width]
        return tensor * 255

    def decode_tensor(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1).contiguous()  # Convert to [batch_size, height, width, channels]
        return tensor / 255
    

    def neural_style_transfer(self, content_img, style_img, content_weight, style_weight, tv_weight, num_steps, learning_rate):

        neural_style_transfer_path = os.path.join(folder_paths.base_path, "custom_nodes/ComfyUI-Fast-Style-Transfer/neural_style_transfer.py")

        temp_save_content_img = os.path.join(folder_paths.base_path, "custom_nodes/ComfyUI-Fast-Style-Transfer/temp/") + "temp_save_content_img.pt"
        temp_save_style_img = os.path.join(folder_paths.base_path, "custom_nodes/ComfyUI-Fast-Style-Transfer/temp/") + "temp_save_style_img.pt"
        
        temp_load_final_img = os.path.join(folder_paths.base_path, "custom_nodes/ComfyUI-Fast-Style-Transfer/temp/") + "temp_load_final_img.pt"

        torch.save(self.encode_tensor(content_img), temp_save_content_img)
        torch.save(self.encode_tensor(style_img), temp_save_style_img)
        

        command = [
            sys.executable, neural_style_transfer_path,
            '--content_weight', str(content_weight),
            '--style_weight', str(style_weight),
            '--tv_weight', str(tv_weight),
            '--temp_save_style_img', temp_save_style_img,
            '--temp_save_content_img', temp_save_content_img,
            '--temp_load_final_img', temp_load_final_img,
            '--num_steps', str(num_steps),
            '--learning_rate', str(learning_rate)
            ]

        sp.run(command)

        image = self.decode_tensor(torch.load(temp_load_final_img))
        os.remove(temp_save_style_img)
        os.remove(temp_save_content_img)
        os.remove(temp_load_final_img)
        return (image,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "FastStyleTransfer": FastStyleTransfer,
    "TrainFastStyleTransfer": TrainFastStyleTransfer,
    "NeuralStyleTransfer": NeuralStyleTransfer,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "FastStyleTransfer": "Fast Style Transfer",
    "TrainFastStyleTransfer": "Train Fast Style Transfer",
    "NeuralStyleTransfer": "Neural Style Transfer",
}
