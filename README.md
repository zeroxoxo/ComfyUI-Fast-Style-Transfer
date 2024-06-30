# ComfyUI-Fast-Style-Transfer
ComfyUI node for fast neural style transfer.

This is a simple conversion based on this:
https://github.com/rrmina/fast-neural-style-pytorch

Only basic inference functionality is ported for now.

![alt text](https://github.com/zeroxoxo/ComfyUI-Fast-Style-Transfer/blob/main/ComfyUI.PNG?raw=true)

If you wanna use custom styles, then clone the original repo and use train.py script, then transfer .pth model file into "ComfyUI/custom_nodes/ComfyUI-Fast-Style-Transfer/models" folder

# Installation

Probably the usual. Just "git clone https://github.com/zeroxoxo/ComfyUI-Fast-Style-Transfer.git" into your custom_nodes folder. That should be it.

If it doesn't work then idk, ask stack exchange or something, how should I know what's wrong with your setup?

# Training

First you'll need to download some files:

VGG-16: https://github.com/jcjohnson/pytorch-vgg

Put it into vgg folder.


MS COCO train dataset.

Original repo suggests train-2014 dataset from here: https://cocodataset.org/#download

But be wary that it's 13Gb.

I used MS COCO train-2017 dataset downscaled to 256x256 from here: https://academictorrents.com/details/eea5a532dd69de7ff93d5d9c579eac55a41cb700

It's only 1.64Gb and original repo still used training with 256x256 size images but it manually downscaled it from the 13Gb dataset.

Put the train-2017 (or train-2014) folder into dataset folder.


That's it for downloads.

Now just use ComfyUI to load TrainFastStyleTransfer node.

To select style picture load "load_image" node, load image inside of it, then press f5, now the image should be in style_img list inside of TrainFastStyleTransfer node, select it.

Adjust batch_size as high as you can with your vram. On my 2060 setup I got 5.9 Gb vram usage running batch_size = 12 ("nvidia-smi" command can be used in cmd to check current vram usage). If you have more you can crank it higher to drastically reduce training time.

One epoch should be fine, but you can test more on your own if your setup is fast enough or you have spare time.

save_model_every will save model and produce test picture every n-th step of training.

After setting all parameters just queue prompt and wait until training is done. Training a model can take up to 2 hours, so have patience.

All intermediate and final models will be saved in models folder, test them, delete redundant and rename the one you like.
