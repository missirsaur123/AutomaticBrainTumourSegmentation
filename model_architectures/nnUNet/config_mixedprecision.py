import torch
import torchio as tio
from torchio.transforms import RescaleIntensity, RandomFlip, RandomAffine, RandomElasticDeformation, RandomGamma

IMAGE_SHAPE = (128,128,128)
NUM_EPOCHS = 1000
BATCH_SIZE = 4
LEARNING_RATE = 3e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SAVE_PATH = 'saved_models/nnUNet/experiment_4'

include_all = ['t1_image', 't2_image', 'flair_image', 't1ce_image', 'seg_mask']
transforms = [
              RandomAffine(scales = (0.7,1.4,0.75,1.5,0.65,1.6), p = 0.3, include = include_all),
              RescaleIntensity(out_min_max = (0.9,1.1), exclude = 'seg_mask'),
              #RandomElasticDeformation(num_control_points = 7, locked_borders= 2, p = 0.3),
              RandomGamma(log_gamma=(-0.3, 0.3), p =0.2, exclude = 'seg_mask'),
              RandomFlip(axes = (0,1,2), flip_probability = 0.5, exclude = 'seg_mask'),
]
aug_transform = tio.Compose(transforms)