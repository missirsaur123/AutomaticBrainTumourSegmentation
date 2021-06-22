import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchio as tio
from torchio.transforms import CropOrPad, ZNormalization, RescaleIntensity
import glob
from sklearn.model_selection import train_test_split
#import config

t1_files = 'data/train_data/*GG/*/*t1.nii.gz'
t2_files = 'data/train_data/*GG/*/*t2.nii.gz'
flair_files = 'data/train_data/*GG/*/*flair.nii.gz'
t1ce_files = 'data/train_data/*GG/*/*t1ce.nii.gz'
seg_files = 'data/train_data/*GG/*/*seg.nii.gz'

t1 = glob.glob(t1_files)
t2 = glob.glob(t2_files)
flair = glob.glob(flair_files)
t1ce = glob.glob(t1ce_files)
seg = glob.glob(seg_files)

data_paths =[]

for items in (zip(t1, t2, flair, t1ce, seg)):
    data_dict = {}
    data_dict['t1'] = items[0]
    data_dict['t2'] = items[1]
    data_dict['flair'] = items[2]
    data_dict['t1ce'] = items[3]
    data_dict['seg'] = items[4]
    data_paths.append(data_dict)

corrupt_files = [0, 159]
correct_data_paths = [i for j,i in enumerate(data_paths) if j not in corrupt_files]

class BRATSDataset(Dataset):
    def __init__(self, data_path, image_shape = (128,128,128), apply_preprocessing = True, transform = None):

        self.data_path = data_path
        self.modalities = ['t1', 't2', 'flair' , 't1ce', 'seg']
        self.image_shape = image_shape
        self.transform = transform
        self.preprocessing_transforms = tio.Compose([
                                                     tio.CropOrPad(target_shape = self.image_shape, mask_name='seg_mask'),
                                                     tio.ZNormalization(masking_method = 'seg_mask', exclude = 'seg_mask')
                                                     #tio.ZNormalization(masking_method = lambda x: x > 0, exclude = 'seg_mask')
                                                     
                                                    ]) 

        self.to_preprocess = apply_preprocessing

    def __len__(self):
        return len(self.data_path)

    @staticmethod
    def preprocess_label(img):
        
        img[img==4] = 3
        #bg = (img==0)
        ncr = (img == 1)  # necrosis 
        ed = (img == 2)  # edema
        et = (img == 3)   # enhancing tissue
        #bg = bg.bool().int()
        ncr = ncr.bool().int()
        ed = ed.bool().int()
        et = et.bool().int()

        #seg_img = torch.cat((bg,ncr, ed, et), axis=0)
        seg_img = torch.cat((ed, ncr, et), axis=0)

        return  seg_img

    def __getitem__(self,idx):

        subject = tio.Subject (
                                t1_image=tio.ScalarImage(self.data_path[idx]['t1']),
                                t2_image=tio.ScalarImage(self.data_path[idx]['t2']),
                                flair_image=tio.ScalarImage(self.data_path[idx]['flair']),
                                t1ce_image=tio.ScalarImage(self.data_path[idx]['t1ce']),
                                seg_mask=tio.LabelMap(self.data_path[idx]['seg']),
                              )

        
        #if self.to_preprocess is True:
        #processed_images = subject
        processed_images = self.preprocessing_transforms(subject)    
        
        if self.transform is not None:

            processed_images = self.transform(processed_images)

        input_image = (torch.cat((processed_images.t1_image.data,
                                  processed_images.t2_image.data,
                                  processed_images.flair_image.data,
                                  processed_images.t1ce_image.data), 
                                  dim = 0)
                                 )
        seg_image = self.preprocess_label(processed_images.seg_mask.data)

        sample = {'input_image': input_image, 'segmentation_image': seg_image}

        return sample


"""transforms = [
              RescaleIntensity(out_min_max = (0.9,1.1), exclude = 'seg_mask'),
              RandomFlip(axes = (0,1,2), flip_probability = 0.5, exclude = 'seg_mask'),
    
]
transform = tio.Compose(transforms)"""

#BRATSDataset = BRATSDataset(data_path = correct_data_paths, transform = transform)

def get_dataloaders(img_shape, transforms, batch_size, split_ratio = 0.9):

    #dataset = BRATSDataset(data_path = correct_data_paths, transform = transform)
    indices = np.arange(len(correct_data_paths))
    train_indices, val_indices = train_test_split(indices, train_size = 0.92, random_state = 25)

    training_file_path = [correct_data_paths[i] for i in train_indices]
    val_file_path = [correct_data_paths[i] for i in val_indices]

    train_dataset = BRATSDataset(data_path = training_file_path, 
                                 image_shape = img_shape, 
                                 transform = transforms
                                 )
    # No transforms are applied to val data
    val_dataset = BRATSDataset(data_path = val_file_path, 
                               image_shape = img_shape, 
                               transform = None
                               )

    train_loader = DataLoader(train_dataset, 
                              shuffle=True, 
                              num_workers=2, 
                              batch_size = batch_size, 
                              pin_memory = False
                              )

    validation_loader = DataLoader(val_dataset, 
                                   shuffle=False, 
                                   num_workers=2, 
                                   batch_size = batch_size
                                   )

    #return (train_loader, validation_loader,train_indices, val_indices)
    return (train_loader, validation_loader)

def get_dataset(img_shape = (128,128,128), transforms = None):

  dataset = BRATSDataset(data_path = correct_data_paths,
                         image_shape = img_shape, 
                         transform = transforms
                         )
  print(len(dataset))
  return dataset
