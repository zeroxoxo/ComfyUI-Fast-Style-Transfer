import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import random
import numpy as np
import time
import argparse
import cv2

parser = argparse.ArgumentParser(description="Train a neural network with style transfer.")

parser.add_argument('--train_image_size', type=int, default=256, help='Size of the training images')
parser.add_argument('--dataset_path', type=str, default="dataset", help='Path to the dataset')
parser.add_argument('--vgg_path', type=str, default="vgg", help='Path to the vgg model')
parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs for training')
parser.add_argument('--temp_save_style_img', type=str, default="temp/temp_save_style_img.pt", help='Path to the style image tensor')
parser.add_argument('--batch_size', type=int, default=12, help='Batch size for training')
parser.add_argument('--content_weight', type=float, default=8, help='Weight for content loss')
parser.add_argument('--style_weight', type=float, default=50, help='Weight for style loss')
parser.add_argument('--tv_weight', type=float, default=0.001, help='Weight for total variation loss')
parser.add_argument('--adam_lr', type=float, default=0.001, help='Learning rate for Adam optimizer')
parser.add_argument('--save_model_path', type=str, default="models/oil/", help='Path to save the trained model')
parser.add_argument('--save_image_path', type=str, default="images/out/", help='Path to save the output images')
parser.add_argument('--save_model_every', type=int, default=200, help='Save model every n batches')
parser.add_argument('--seed', type=int, default=1234, help='Random seed')
parser.add_argument('--pretrained_model', type=str, default='none', help='pretrained model')

args = parser.parse_args()

# GLOBAL SETTINGS
TRAIN_IMAGE_SIZE = args.train_image_size
DATASET_PATH = args.dataset_path
NUM_EPOCHS = args.num_epochs
STYLE_IMAGE_PATH = args.temp_save_style_img
VGG_PATH = args.vgg_path
BATCH_SIZE = args.batch_size
CONTENT_WEIGHT = args.content_weight
STYLE_WEIGHT = args.style_weight
TV_WEIGHT = args.tv_weight
ADAM_LR = args.adam_lr
SAVE_MODEL_PATH = args.save_model_path
SAVE_IMAGE_PATH = args.save_image_path
SAVE_MODEL_EVERY = args.save_model_every
SEED = args.seed
MODEL = args.pretrained_model

# Utils
# Gram Matrix
def gram(tensor):
    B, C, H, W = tensor.shape
    x = tensor.view(B, C, H*W)
    x_t = x.transpose(1, 2)
    gram_matrix = torch.bmm(x, x_t) / (C*H*W)
    gram_matrix = torch.clamp(gram_matrix, min=1e-6, max=1e6)
    return gram_matrix

# Save image
def saveimg(img, image_path):
    img = img.clip(0, 255)
    cv2.imwrite(image_path, img)

# Preprocessing ~ Tensor to Image
def ttoi(tensor):
    # Add the means
    #ttoi_t = transforms.Compose([
    #    transforms.Normalize([-103.939, -116.779, -123.68],[1,1,1])])

    # Remove the batch_size dimension
    tensor = tensor.squeeze()
    #img = ttoi_t(tensor)
    img = tensor.cpu().numpy()
    
    # Transpose from [C, H, W] -> [H, W, C]
    img = img.transpose(1, 2, 0)
    return img

# Alternative loss functions experiments
def total_variation(y):
    return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + \
           torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))

# VGG
class VGG16(nn.Module):
    def __init__(self, vgg_path="models/vgg16-00b39a1b.pth"):
        super(VGG16, self).__init__()
        # Load VGG Skeleton, Pretrained Weights
        vgg16_features = models.vgg16(pretrained=False)
        vgg16_features.load_state_dict(torch.load(vgg_path), strict=False)
        self.features = vgg16_features.features

        # Turn-off Gradient History
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        layers = {'3': 'relu1_2', '8': 'relu2_2', '15': 'relu3_3', '22': 'relu4_3'}
        features = {}
        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
                if (name=='22'):
                    break

        return features

# Transformer
class TransformerNetworkClass(nn.Module):
    """Feedforward Transformation Network without Tanh
    reference: https://arxiv.org/abs/1603.08155 
    exact architecture: https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf
    """
    def __init__(self):
        super(TransformerNetworkClass, self).__init__()
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

class ConvolutionalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm="instance"):
        super(ConvolutionalLayer, self).__init__()
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


def train():

    # Seeds
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Device
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Train.py: Device is {device}")

    # Dataset and Dataloader
    transform = transforms.Compose([
        transforms.Resize(TRAIN_IMAGE_SIZE),
        transforms.CenterCrop(TRAIN_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Train.py: train_loader length: {len(train_loader)}")

    # Load networks
    TransformerNetwork = TransformerNetworkClass().to(device)
    if MODEL != 'none':
        model_path = SAVE_MODEL_PATH + MODEL
        print(f"Loading {model_path} model")
        TransformerNetwork.load_state_dict(torch.load(model_path, map_location=device))
        TransformerNetwork.to(device)
    VGG = VGG16(vgg_path=VGG_PATH).to(device)

    # Get Style Features
    imagenet_neg_mean = torch.tensor([-103.939, -116.779, -123.68], dtype=torch.float32).reshape(1,3,1,1).to(device)
    style_tensor = torch.load(STYLE_IMAGE_PATH).to(device)
    style_tensor = style_tensor.add(imagenet_neg_mean)
    B, C, H, W = style_tensor.shape
    style_features = VGG(style_tensor.expand([BATCH_SIZE, C, H, W]))
    style_gram = {}
    for key, value in style_features.items():
        style_gram[key] = gram(value)

    # Optimizer settings
    optimizer = optim.AdamW(TransformerNetwork.parameters(), lr=ADAM_LR, fused=True, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6, last_epoch=-1)
    # Loss trackers
    content_loss_history = []
    style_loss_history = []
    total_loss_history = []
    batch_content_loss_sum = 0
    batch_style_loss_sum = 0
    batch_total_loss_sum = 0
    # Optimization/Training Loop
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        print("========Epoch {}/{}========".format(epoch+1, NUM_EPOCHS))
        for batch_count, (content_batch, _) in enumerate(train_loader):
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            # Generate images and get features
            content_batch = content_batch[:,[2,1,0]].to(device)
            generated_batch = TransformerNetwork(content_batch)
            content_features = VGG(content_batch.add(imagenet_neg_mean))
            generated_features = VGG(generated_batch.add(imagenet_neg_mean))

            # Content Loss
            MSELoss = nn.MSELoss().to(device)
            MAELoss = nn.L1Loss().to(device)
            content_loss = CONTENT_WEIGHT * MSELoss(generated_features['relu2_2'], content_features['relu2_2']) + MAELoss(generated_features['relu2_2'], content_features['relu2_2']) / 2
            batch_content_loss_sum += content_loss

            # Style Loss
            style_loss = 0.0
            for key, value in generated_features.items():
                style_loss += MSELoss(gram(value), style_gram[key][:content_batch.shape[0]])
                style_loss += MAELoss(gram(value), style_gram[key][:content_batch.shape[0]])
            style_loss *= STYLE_WEIGHT/2
            batch_style_loss_sum += style_loss.item()

            # TV loss
            tv_loss = total_variation(generated_batch) * TV_WEIGHT

            # Total Loss
            total_loss = content_loss + style_loss + tv_loss
            batch_total_loss_sum += total_loss.item()
            
            total_loss.backward()
            optimizer.step()
            if batch_count % 50 == 0:
                scheduler.step()

            with torch.no_grad():
                print(f'AdamW | iteration: {batch_count+1:03}, total loss={total_loss.item():12.4f}, content_loss={content_loss.item():12.4f}, style loss={style_loss.item():12.4f}, tv loss={tv_loss.item():12.4f}')

            # Save Model and Print Losses
            if (((batch_count)%SAVE_MODEL_EVERY == 0) or ((batch_count+1)==NUM_EPOCHS*len(train_loader))):
                # Print Losses
                print("========Iteration {}/{}========".format(batch_count+1, NUM_EPOCHS*len(train_loader)))
                print("\tContent Loss:\t{:.2f}".format(batch_content_loss_sum/(batch_count+1)))
                print("\tStyle Loss:\t{:.2f}".format(batch_style_loss_sum/(batch_count+1)))
                print("\tTotal Loss:\t{:.2f}".format(batch_total_loss_sum/(batch_count+1)))
                print("Time elapsed:\t{} seconds".format(time.time()-start_time))

                # Save Model
                checkpoint_path = SAVE_MODEL_PATH + "checkpoint_" + str((batch_count+1)) + ".pth"
                torch.save(TransformerNetwork.state_dict(), checkpoint_path)
                print("Saved TransformerNetwork checkpoint file at {}".format(checkpoint_path))

                # Save sample generated image
                sample_tensor = generated_batch[0].clone().detach().unsqueeze(dim=0)
                sample_image = ttoi(sample_tensor.clone().detach())
                sample_image_path = SAVE_IMAGE_PATH + "sample0_" + str((batch_count+1)) + ".png"
                saveimg(sample_image, sample_image_path)
                print("Saved sample tranformed image at {}".format(sample_image_path))

                # Save loss histories
                content_loss_history.append(batch_content_loss_sum/(batch_count+1))
                style_loss_history.append(batch_style_loss_sum/(batch_count+1))
                total_loss_history.append(batch_total_loss_sum/(batch_count+1))

            

                


    stop_time = time.time()
    # Print loss histories
    print("Done Training the Transformer Network!")
    print("Training Time: {} seconds".format(stop_time-start_time))
    print("========Content Loss========")
    print(content_loss_history) 
    print("========Style Loss========")
    print(style_loss_history) 
    print("========Total Loss========")
    print(total_loss_history) 


if __name__ == "__main__":
    train()

