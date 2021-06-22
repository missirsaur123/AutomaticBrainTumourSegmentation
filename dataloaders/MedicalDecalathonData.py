import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchio as tio
from torchio.transforms import CropOrPad, ZNormalization
import glob

input_path = 'data/medicaldecalathon/Task01_BrainTumour/imagesTr/BRATS_*.nii.gz'
label_path = 'data/medicaldecalathon/Task01_BrainTumour/labelsTr/BRATS_*.nii.gz'

input_files = sorted(glob.glob(input_path))
label_files = sorted(glob.glob(label_path))

file_paths = list(zip(input_files,label_files))

class MedicalDecalathonDataset(Dataset):

    def __init__(self, data_path, image_shape = (128,128,128), transform = None):

        self.data_path = data_path
        self.modalities = ['t1', 't2', 'flair' , 't1ce', 'seg']
        self.image_shape = image_shape
        self.preprocessing_transforms = tio.Compose([
                                                     CropOrPad(target_shape = self.image_shape, mask_name='seg_mask'),
                                                     ZNormalization(masking_method = 'seg_mask', exclude = 'seg_mask')]) 

        self.transform = transform

    def __len__(self):
        return len(self.data_path)

    @staticmethod
    def preprocess_label(img):


        # 1 for NCR & NET, 2 for ED, 4 for ET, and 0 for everything else
        #bg = (img==0)
        ncr = (img == 1)  # necrosis 
        ed = (img == 2)  # edema
        et = (img == 3)   # enhancing tissue
        #bg = bg.bool().int()
        ncr = ncr.bool().int()
        ed = ed.bool().int()
        et = et.bool().int()

        #seg_img = torch.cat((bg,ncr, ed, et), axis=0)
        seg_img = torch.cat((ncr, ed, et), axis=0)

        return  seg_img

    def __getitem__(self,idx):

        image_data_path, label_data_path = self.data_path[idx]

        image_data = tio.ScalarImage(image_data_path)
        label_image = tio.LabelMap(label_data_path)

        # Order of MRI modalities has been fixated to : t1,t2,flair,t1ce
        """patient_data = tio.Subject(          
                                    t1_image = tio.ScalarImage(tensor = torch.unsqueeze(image_data.data[:,:,:,0], dim = 0)),
                                    t2_image =  tio.ScalarImage(tensor = torch.unsqueeze(image_data.data[:,:,:,2], dim = 0)),
                                    flair_image = tio.ScalarImage(tensor = torch.unsqueeze(image_data.data[:,:,:,3], dim = 0)),
                                    t1ce_image =  tio.ScalarImage(tensor = torch.unsqueeze(image_data.data[:,:,:,1], dim = 0)),
                                    seg_mask = label_image
                                  )"""

        patient_data = tio.Subject(          
                                    t1_image = tio.ScalarImage(tensor = image_data.data[:,:,:,0].unsqueeze_(0)),
                                    t2_image = tio.ScalarImage(tensor = image_data.data[:,:,:,2].unsqueeze_(0)),
                                    flair_image = tio.ScalarImage(tensor = image_data.data[:,:,:,3].unsqueeze_(0)),
                                    t1ce_image = tio.ScalarImage(tensor = image_data.data[:,:,:,1].unsqueeze_(0)),
                                    seg_mask = label_image
                                  )
        
        #del image_data

        processed_images = self.preprocessing_transforms(patient_data)

        if self.transform is not None:
            processed_images = self.transform(processed_images)

        input_image = (torch.cat((processed_images.t1_image.data,processed_images.t2_image.data,processed_images.flair_image.data,processed_images.t1ce_image.data), dim = 0))
        seg_image = self.preprocess_label(processed_images.seg_mask.data)

        sample = {'input_image': input_image, 'segmentation_image': seg_image}
        
        return sample


def get_dataloaders(img_shape, transforms, batch_size, split_ratio = 0.9):

    #dataset = MedicalDecalathonDataset(data_path = file_paths,image_shape = img_shape, transform = transforms)
    indices = np.arange(len(file_paths))
    train_indices, val_indices = train_test_split(indices, train_size = 0.92, random_state = 42)

    training_file_path = [file_paths[i] for i in train_indices]
    val_file_path = [file_paths[i] for i in val_indices]

    train_dataset =  MedicalDecalathonDataset(data_path = training_file_path,
                                              image_shape = img_shape, 
                                              transform = transforms)
    # validation dataset should not have transforms
    validation_dataset = MedicalDecalathonDataset(data_path = val_file_path,
                                                  image_shape = img_shape,
                                                  transform = None)
    

    train_loader = DataLoader(train_dataset, 
                              batch_size = batch_size, 
                              shuffle = True, 
                              num_workers = 2,
                              pin_memory = True)
    validation_loader = DataLoader(validation_dataset,
                                   batch_size = batch_size, 
                                   shuffle = False, 
                                   num_workers=2)

    return (train_loader, validation_loader)

def get_dataset(img_shape = (128,128,128), transforms = None):

  dataset = MedicalDecalathonDataset(data_path = file_paths,
                         image_shape = img_shape, 
                         transform = transforms
                         )
  print(len(dataset))
  return dataset