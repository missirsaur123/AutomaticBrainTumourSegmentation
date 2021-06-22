import numpy as np
import os
import glob
import time
from tqdm import tqdm
import gc
import pickle
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataloaders.BRATS2018Data import get_dataloaders
from model_architectures.nnUNet.nnUNet import nnUNetModel
import model_architectures.nnUNet.config_nnunet as config
from losses import BCE_plus_Dice_Loss, MultiClassDiceLoss
from metrics import DiceScore


model = nnUNetModel().to(config.device)

train_dataloader, validation_dataloader = get_dataloaders(img_shape = config.IMAGE_SHAPE, batch_size = config.BATCH_SIZE, transforms = config.aug_transform)

#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.99, weight_decay = 0.001)
#adw_optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
adw_optimizer = optim.Adam(model.parameters(), lr = 3e-4)
#lr_lambda = lambda epoch : (1 - epoch/config.NUM_EPOCHS)**0.96
#scheduler = LambdaLR(optimizer, lr_lambda = lr_lambda)

scheduler = ReduceLROnPlateau(adw_optimizer, mode='max', factor=0.3, patience=3, verbose=True)
#scheduler = torch.optim.lr_scheduler.OneCycleLR(adw_optimizer, max_lr=0.1, steps_per_epoch=len(train_dataloader),epochs = 200, pct_start = 0.6)
#scaler = torch.cuda.amp.GradScaler(init_scale=2**18 ,enabled=True)

def train(model, train_dataloader, optimizer,device = 'cuda'):

  train_loss = []
  train_acc = []
  #tbdata = next(iter(train_dataloader))
  #for i in range(1):
  for tbdata in tqdm(train_dataloader):
    
    gc.collect()
    model.train()

    input = tbdata['input_image']
    label = tbdata['segmentation_image']
    input = input.to(device = device, dtype = torch.float32)
    label = label.to(device = device, dtype = torch.float32)
    optimizer.zero_grad()

    output = model(input)
    loss = BCE_plus_Dice_Loss(output, label)
    t_acc = DiceScore(output,label)
    loss.backward()
    optimizer.step()
    
    train_loss.append(loss.detach().item())
    train_acc.append(t_acc.detach().item())

    del input
    del label
    del output
    torch.cuda.empty_cache()

  mean_train_loss = sum(train_loss)/len(train_loss)
  mean_train_acc = sum(train_acc)/len(train_acc)
    
  return mean_train_loss, mean_train_acc

def validate(model, val_dataloader, device = 'cuda'):
  
  val_acc = []
  #vbdata = next(iter(val_dataloader))
  with torch.no_grad():
    model.eval()
    for vbdata in tqdm(val_dataloader): 

        input = vbdata['input_image']
        label = vbdata['segmentation_image']
        input = input.to(device = device, dtype = torch.float32)
        label = label.to(device = device, dtype = torch.float32)

        output = model(input)
        v_acc = DiceScore(output, label)

        val_acc.append(v_acc.detach().item())
    
  mean_val_acc = sum(val_acc)/len(val_acc)

  return mean_val_acc

def save_metrics(epoch, mean_train_loss,mean_train_acc, mean_val_acc):

  training_info_list = []
  epoch_info_dict = {epoch : [mean_train_loss,mean_train_acc, mean_val_acc]}

  with open('saved_models/nnUNet/experiment_1/training_info.pkl', 'rb') as fp:
    try:
      training_info_list = pickle.load(fp)
    except:
      training_info_list = []
      
  with open('saved_models/nnUNet/experiment_4/training_info.pkl', 'wb') as fp:
    training_info_list.append(epoch_info_dict)
    pickle.dump(training_info_list, fp)


def run(start_epoch,
        total_epochs, 
        model, 
        train_dataloader, 
        val_dataloader, 
        optimizer, 
        scheduler, 
        save_path,
        device):
  

  print(optimizer)
  for epoch in range(start_epoch, total_epochs):
    
    start = time.time()
    print('\nEpoch {}/{}'.format(epoch, total_epochs))

    mean_train_loss, mean_train_acc = train(model, train_dataloader, optimizer)
    mean_val_acc = validate(model, val_dataloader)

    print(f"training loss : {(mean_train_loss):.4f}, training accuracy : {(mean_train_acc):.4f}")
    print(f"validation accuracy : {(mean_val_acc):.4f}")

    scheduler.step(mean_train_acc)
  
    save_metrics(epoch, mean_train_loss,mean_train_acc, mean_val_acc)

    #to_save = True if (epoch < 40 and epoch % 5==1) or (epoch >40 and epoch %10==9) else False
    to_save = (epoch % 5==1) 

    if to_save is True:
      state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        'scheduler_dict': scheduler.state_dict(),
        #'scaler_dict': scaler.state_dict(),
        'train_loss' : mean_train_loss,
        'val_acc' : mean_val_acc
              }

      ckpt_name = f'model_ckpt_{epoch}.pt'
      model_ckpt_path = os.path.join(save_path,ckpt_name)
      torch.save(state,model_ckpt_path)

    end = time.time()
    minutes, seconds = divmod(end-start, 60)     # try to improve time formatiting
    minutes = int(minutes)
    seconds = int(seconds)
    print(f"Epoch {epoch} ran {minutes}::{seconds}")


def resume_training_from_checkpoint(path):

  checkpoint = torch.load(path)

  ckpt_model = nnUNetModel().to(config.device)
  #ckpt_optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=0.99, weight_decay = 1e-5)
  ckpt_optimizer = adw_optimizer
  ckpt_scheduler = scheduler
  #ckpt_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=3, verbose=True)

  ckpt_model.load_state_dict(checkpoint['state_dict'])
  #ckpt_optimizer.load_state_dict(checkpoint['optimizer_dict'])
  #ckpt_scheduler.load_state_dict(checkpoint['scheduler_dict'])
  #ckpt_scaler.load_state_dict(checkpoint['scaler_dict'])
  epoch = checkpoint['epoch']

  run(start_epoch = epoch+1, 
      total_epochs = 300, 
      model = ckpt_model, 
      train_dataloader = train_dataloader, 
      val_dataloader = validation_dataloader, 
      optimizer = ckpt_optimizer, 
      scheduler = ckpt_scheduler,
      save_path = config.MODEL_SAVE_PATH,
      device = config.device)

RESUME_TRAINING = True

if RESUME_TRAINING is True:

  files_list = glob.glob(config.MODEL_SAVE_PATH + '/*')
  ckpt_path = max(files_list, key=os.path.getctime)
  #ckpt_path = '/content/drive/MyDrive/AutomaticBrainTumorSegmentation/saved_models/nnUNet/experiment_2/model_ckpt_1.pt'
  resume_training_from_checkpoint(ckpt_path)
  
else:
  
  run(start_epoch = 1, 
      total_epochs = config.NUM_EPOCHS, 
      model = model, 
      train_dataloader = train_dataloader, 
      val_dataloader = validation_dataloader, 
      optimizer = optimizer, 
      scheduler = scheduler,
      save_path = config.MODEL_SAVE_PATH,
      device = config.device)