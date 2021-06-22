import torch
import torchio as tio
from torchio.transforms import RescaleIntensity, RandomFlip, RandomAffine, RandomElasticDeformation, RandomGamma


# random rotations, randomscaling, random elastic deformations, gamma correction augmentation and mirroring.
""" 
increase the probability of applying rotation and scaling from 0.2 to 0.3.
increase the scale range from (0.85, 1.25) to (0.65, 1.6)
select a scaling factor for each axis individually
use elastic deformation with a probability of 0.3
use additive brightness augmentation with a probability of 0.3
increase the aggressiveness of the Gamma augmentation """

IMAGE_SHAPE = (128,128,128)
NUM_EPOCHS = 1000
BATCH_SIZE = 2
LEARNING_RATE = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SAVE_PATH = 'saved_models/nnUNet/experiment_2'

include_all = ['t1_image', 't2_image', 'flair_image', 't1ce_image', 'seg_mask']
transforms = [
              RandomAffine(scales = (0.7,1.4,0.75,1.5,0.65,1.6), p = 0.3, include = include_all),
              RescaleIntensity(percentiles = (1,99), exclude = 'seg_mask'),
              #RandomElasticDeformation(num_control_points = 7, locked_borders= 2, p = 0.3),
              RandomGamma(log_gamma=(-0.3, 0.3), p =0.2, exclude = 'seg_mask'),
              RandomFlip(axes = (0,1,2), flip_probability = 0.5, exclude = 'seg_mask'),
]
aug_transform = tio.Compose(transforms)
