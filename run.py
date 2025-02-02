import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Only ignore specific warning types
warnings.filterwarnings('ignore', category=Warning, message=".*libpng.*")

from data.datasets import InterpolatedImageDataset
from training.train import Train
from arch.unet import UNet
import torch

dataset = InterpolatedImageDataset.from_hf('datasets/laion_hf', 'images', None)
config = {
    "dims": [64, 128, 256, 512],
    "in_channels" : 3,
    "out_channels" : 3,
    "kernel_size" : 3,
    "stride" : 1,
    "padding" : "same",
    "dilation" : 1,
    "num_conv" : 2,
    "image_max_size": 512
}
model = UNet(config).to('cuda', dtype=torch.bfloat16)

Train(model,
    dataset,
    run_name="unet_0101", 
    batch_size=1, 
    grad_accum_steps=8,
    lr=1e-3,
    loss_fn='mse',
    save_steps=5,
    seed = 111,
    num_workers=4,
    warmup_steps=100,
    validation = False, max_saves = 1, max_steps=None, save_dir="results", resume_from="latest")
