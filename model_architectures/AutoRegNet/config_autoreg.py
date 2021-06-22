import torch
import torchio
from torchio.transforms import CropOrPad, ZNormalization, RescaleIntensity, RandomFlip

IMAGE_SHAPE = (160,192,128)
NUM_EPOCHS = 300
BATCH_SIZE = 1
LEARNING_RATE = 3e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SAVE_PATH = 'models/AutoRegNet/experiment1'

transforms = [
              RescaleIntensity(out_min_max = (0.9,1.1), exclude = 'seg_mask'),
              RandomFlip(axes = (0,1,2), flip_probability = 0.5, exclude = 'seg_mask'),
    
]
transform = tio.Compose(transforms)
