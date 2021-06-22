import numpy as np
import os
import time
from tqdm import tqdm
import gc
import copy
import pickle
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import LambdaLR

from dataloaders.BRATS2018Data import get_dataloaders
from model_architectures.nnUNet.ResUNet_drop import ResUNetModel
from losses import Weighted_CE_plus_Dice_Loss
import model_architectures.nnUNet.config_nnunet as config
from metrics import DiceScore, ClassDiceScore

model = ResUNetModel().to(config.device)

def init_weights(m):
    if type(m) == nn.Conv3d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


train_dataloader, validation_dataloader = get_dataloaders(img_shape = config.IMAGE_SHAPE, batch_size = config.BATCH_SIZE, transforms = config.aug_transform)

#optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=0.99, weight_decay = 0.001)
optimizer = optim.Adam(model.parameters(), lr = 1e-4, weight_decay = 1e-4)
#optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

#lr_lambda = lambda epoch : (1 - epoch/config.NUM_EPOCHS)**0.96
#scheduler = LambdaLR(optimizer, lr_lambda = lr_lambda)

#scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_dataloader), epochs=1000)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor = 0.5, patience = 10, verbose=True)

SAVE_PATH = 'saved_models/nnUNet/experiment_6'
#scaler = torch.cuda.amp.GradScaler(init_scale=2**18 ,enabled=False)
torch.backends.cudnn.benchmark = True

def save_metrics(metrics_list):

  training_info_list = []
  #epoch_info_dict = {epoch : [mean_train_loss,mean_train_acc, mean_val_acc]}
  info_file_path = os.path.join(SAVE_PATH, 'training_info.pkl')
  
  with open(info_file_path, 'rb') as fp:
    try:
      training_info_list = pickle.load(fp)
    except:
      training_info_list = []
  
  with open(info_file_path, 'wb') as fp:
    for info in metrics_list:
      training_info_list.append(info)
    pickle.dump(training_info_list, fp)

def run(start_epoch, 
        model, 
        train_dataloader, 
        val_dataloader, 
        optimizer, 
        scheduler, 
        total_epochs):
  
  for epoch in range(start_epoch, total_epochs):
        
    start = time.time()
    print('\nEpoch {}/{}'.format(epoch, total_epochs))

    train_loss = []
    train_acc = []
    val_acc = []
    class_dice_scores = [] 
    metrics_to_save = []
    
    #tbdata = next(iter(train_dataloader))
    #print(len(da))
    for tbdata in tqdm(train_dataloader):
    #for i in range(1):
      
      #gc.collect()
      model.train()

      input = tbdata['input_image']
      label = tbdata['segmentation_image']
      input = input.to(device = config.device, dtype = torch.float32)
      label = label.to(device = config.device, dtype = torch.float32)
      
      optimizer.zero_grad()

      output = model(input)
      
      loss = Weighted_CE_plus_Dice_Loss(output, label)
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
    
    #vbdata = next(iter(val_dataloader))
    model.eval()
    with torch.no_grad():
      for vbdata in tqdm(val_dataloader):
      #for i in range(1):

          input = vbdata['input_image']
          label = vbdata['segmentation_image']
          input = input.to(device = config.device, dtype = torch.float32)
          label = label.to(device = config.device, dtype = torch.float32)

          output = model(input)
          v_acc = DiceScore(output, label)
          class_dice_scores.append(ClassDiceScore(output,label))
        
          val_acc.append(v_acc.detach().item())

          del input
          del label
          del output
          torch.cuda.empty_cache()
      
    mean_val_acc = sum(val_acc)/len(val_acc)

    print(f"training loss : {(mean_train_loss):.4f}, training accuracy : {(mean_train_acc):.4f}")
    print(f"validation accuracy : {(mean_val_acc):.4f}")

    a,b,c = [],[],[]
    for x in class_dice_scores:
      a.append(x[0].item())
      b.append(x[1].item())
      c.append(x[2].item())

    print(sum(a)/len(a), sum(b)/len(b), sum(c)/len(c))

    metrics_to_save.append({epoch : [mean_train_loss,mean_train_acc, mean_val_acc]})

    scheduler.step(mean_train_loss)
    #scheduler.step()

    #save_metrics(epoch, mean_train_loss,mean_train_acc, mean_val_acc)

    to_save = epoch % 5 == 2 
    if to_save is True:
      state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        'scheduler_dict': scheduler.state_dict(),
        'train_loss' : mean_train_loss,
        'val_acc' : mean_val_acc
      }

      ckpt_name = f'model_ckpt_{epoch}.pt'
      savepath = os.path.join(SAVE_PATH,ckpt_name)
      torch.save(state,savepath)
      save_metrics(metrics_to_save)
      print(optimizer)

    end = time.time()
    minutes, seconds = divmod(end-start, 60)     # try to improve time formatiting
    minutes = int(minutes)
    seconds = int(seconds)
    print(f"Epoch {epoch} ran {minutes}::{seconds}")


def resume_training_from_checkpoint(path):

  #import model_architectures.nnUNet.config_mixedprecision as config

  checkpoint = torch.load(path)

  ckpt_model = model
  ckpt_optimizer = copy.deepcopy(optimizer)
  ckpt_scheduler = copy.deepcopy(scheduler)
  ckpt_optimizer = optimizer
  ckpt_scheduler = scheduler

  #ckpt_scaler = scaler

  ckpt_model.load_state_dict(checkpoint['state_dict'])
  #ckpt_optimizer.load_state_dict(checkpoint['optimizer_dict'])
  #ckpt_scheduler.load_state_dict(checkpoint['scheduler_dict'])
  #ckpt_scaler.load_state_dict(checkpoint['scaler_dict'])
  epoch = checkpoint['epoch']

  print(ckpt_optimizer)
  print(ckpt_scheduler)

  run(start_epoch = epoch + 1, 
      total_epochs = 1000, 
      model = ckpt_model, 
      train_dataloader = train_dataloader, 
      val_dataloader = validation_dataloader, 
      optimizer = ckpt_optimizer, 
      scheduler = ckpt_scheduler)

RESUME_TRAINING = True

if RESUME_TRAINING is True:
  files_list = glob.glob(SAVE_PATH + '/*.pt')
  ckpt_path = max(files_list, key=os.path.getctime)
  
  #ckpt_path = '/content/drive/MyDrive/AutomaticBrainTumorSegmentation/saved_models/nnUNet/experiment_1/model_ckpt_12.pt'
  print(ckpt_path)
  resume_training_from_checkpoint(ckpt_path)
  
else:
  print(optimizer)
  model.apply(init_weights)
  run(start_epoch = 1, 
      total_epochs = config.NUM_EPOCHS, 
      model = model, 
      train_dataloader = train_dataloader, 
      val_dataloader = validation_dataloader, 
      optimizer = optimizer, 
      scheduler = scheduler)