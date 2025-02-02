import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseArch
from .blocks import MultiConv
import gc
from PIL import Image
import cv2

default_config = {
    "dims": [64, 128, 256, 512],
    "in_channels" : 3,
    "out_channels" : 3,
    "kernel_size" : 3,
    "stride" : 1,
    "padding" : "same",
    "dilation" : 1,
    "num_conv" : 2,
    "image_max_size": 512,
}

class UNet(BaseArch):
    def __init__(self, config = default_config):
        super(UNet, self).__init__(config)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.bfloat16 

        dims = self.config["dims"]
        in_channels = self.config["in_channels"]
        out_channels = self.config["out_channels"]
        kernel_size = self.config["kernel_size"]
        padding = self.config["padding"]
        if padding == "same":
            self.config["stride"] = 1
        stride = self.config["stride"]
        dilation = self.config["dilation"]
        num_conv = self.config["num_conv"]
        self.image_max_size = self.config["image_max_size"]

        for idx, dim in enumerate(dims):
            if dim != dims[-1]:
                assert any([dims[idx + 1] == dim * 2, dims[idx + 1] == dim/2]), "UNet layer dimension must be 2x or .5x of the previous layer"

        # Init encoders
        self.encoders = nn.ModuleList([MultiConv(in_channels, dims[0], num_conv, kernel_size, stride, padding)])
        for idx, dim in enumerate(dims):
            if dim != dims[-1]:
                self.encoders.append(MultiConv(dims[idx], dims[idx+1], num_conv, kernel_size, stride, padding))
        self.bottleneck = nn.ModuleList([
            MultiConv(dims[-1], dims[-1] * 2, 3, kernel_size, stride, padding),
            # MultiConv(dims[-1] * 2, dims[-1] * 2),
            # MultiConv(dims[-1] * 2, dims[-1] * 2),
            # MultiConv(dims[-1] * 2, dims[-1] * 2),
            ])
            
        # Init decoders
        revdims = [i for i in reversed(dims)]
        self.decoders = nn.ModuleList([MultiConv(revdims[0], revdims[0], num_conv, kernel_size, stride, padding, is_unet_decoder=True)])
        for idx, dim in enumerate(revdims):
            if dim != revdims[0]:
                self.decoders.append(MultiConv(revdims[idx], revdims[idx], num_conv, kernel_size, stride, padding, is_unet_decoder=True))
        # Init output
        self.output = nn.ModuleList([
            nn.Conv2d(revdims[-1], out_channels, kernel_size, stride, padding)
            ])
        
        print(self)
        self = self.to(self.device).to(dtype=self.dtype)
        self.check()
        print(self.num_params())

    def forward(self, x):
        input_size = x.size()
        encoder_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoder_features.append(x)
            x = nn.MaxPool2d(2)(x)
        
        for bottleneck_layer in self.bottleneck:
            x = bottleneck_layer(x)
        
        encoder_features.reverse()
        for idx, decoder in enumerate(self.decoders):
            x = self.up_concat(x, encoder_features[idx])
            x = decoder(x)
            gc.collect()
        
        for layer in self.output:
            x = layer(x)        
        if x.size() != input_size:
            x = x[:, :, :input_size[2], :input_size[3]]
        return x

    def up_concat(self, x, encoder_feature):
        x = F.interpolate(x, size=encoder_feature.shape[2:], mode='nearest')
        return torch.cat([x, encoder_feature], dim=1)

    def check(self):
        try:
            x = torch.randn(1, self.config["in_channels"], 256, 256).to(self.device).to(self.dtype)
            y = self.forward(x)
            assert x.size() == y.size(), f"Model forward method is NOT working with dummy input size: {x.size()}, mismatch with output size {y.size()}"
            print(f"Model forward method is working with dummy input size: {x.size()}")
        except Exception as e:
            print(f"Error checking forward method with input of size {x.size()}")
            raise
            
    def resize_image(self, image):
        target_size = (self.image_max_size, self.image_max_size)
        h, w = image.shape[:2]
        ratio = min(target_size[0]/w, target_size[1]/h)
        new_size = (int(w * ratio), int(h * ratio))
        
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)
        return image

    def load_image(self, path):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.resize_image(image)
        image = torch.from_numpy(np.array(image)).permute(2,0,1).to(torch.bfloat16)/255.0
        return image

    @torch.no_grad()
    def inference(self, input_path, output_path):
        assert isinstance(input_path, str)
        image = self.load_image(input_path)
        image = image.unsqueeze(0).to(self.device).to(self.dtype)
        output = self.forward(image)
        Image.fromarray((output.squeeze().permute(1,2,0).float().cpu().numpy()*255).astype(np.uint8)).save(f'{output_path}.jpg')
