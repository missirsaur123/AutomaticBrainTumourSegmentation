import numpy as np
import os
import time
from tqdm import tqdm
import gc
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataloaders.MedicalDecalathonData import get_dataloaders
from model_architectures.nnUNet.nnUNet import nnUNetModel
from losses import BCE_plus_Dice_Loss, MultiClassDiceLoss
import model_architectures.nnUNet.config_mixedprecision as config
from metrics import DiceScore
import pickle

model = nnUNetModel().to(config.device)

train_dataloader, validation_dataloader = get_dataloaders(img_shape = config.IMAGE_SHAPE, batch_size = config.BATCH_SIZE, transforms = config.aug_transform)

#optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=0.99, weight_decay = 0.001)
adw_optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.99), weight_decay=0.0001)
#optimizer = optim.Adam(model.parameters(), lr = config.LEARNING_RATE, weight_decay = 1e-5)
#lr_lambda = lambda epoch : (1 - epoch/config.NUM_EPOCHS)**0.96
#scheduler = LambdaLR(optimizer, lr_lambda = lr_lambda)
scheduler = torch.optim.lr_scheduler.OneCycleLR(adw_optimizer, max_lr=0.1, steps_per_epoch=len(train_dataloader), epochs=100)
#scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=3, verbose=True)
scaler = torch.cuda.amp.GradScaler(init_scale=2**18 ,enabled=True)

def save_metrics(epoch, mean_train_loss,mean_train_acc, mean_val_acc):

  training_info_list = []
  epoch_info_dict = {epoch : [mean_train_loss,mean_train_acc, mean_val_acc]}
  with open('saved_models/nnUNet/experiment_4/training_info.pkl', 'rb') as fp:
    try:
      training_info_list = pickle.load(fp)
    except:
      training_info_list = []
  with open('saved_models/nnUNet/experiment_4/training_info.pkl', 'wb') as fp:
    training_info_list.append(epoch_info_dict)
    pickle.dump(training_info_list, fp)

def run(start_epoch, model, train_dataloader, val_dataloader, optimizer, scheduler, total_epochs = 20):
  
  for epoch in range(start_epoch, total_epochs):
    
    start = time.time()
    print('\nEpoch {}/{}'.format(epoch, total_epochs))

    train_loss = []
    train_acc = []
    val_acc = []
    #tbdata = next(iter(train_dataloader))
    #print(len(da))
    for tbdata in tqdm(train_dataloader):
    #for i in range(1):
      
      #gc.collect()
      model.train()

      input = tbdata['input_image']
      label = tbdata['segmentation_image']
      input = input.to(device = config.device, dtype = torch.float16)
      label = label.to(device = config.device, dtype = torch.float16)
      optimizer.zero_grad()

      with torch.cuda.amp.autocast(enabled=True):
        output = model(input)
        
        loss = BCE_plus_Dice_Loss(output, label)
        t_acc = DiceScore(output,label)
        
        assert loss.dtype is torch.float32
        
      #t_acc = DiceScore(output,label)
      scaler.scale(loss).backward()
      scaler.scale(t_acc)
      # check if this part improves accuracy
      scaler.unscale_(optimizer)
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)

      scaler.step(optimizer)
      scaler.update()

      train_loss.append(loss.detach().item())
      train_acc.append(t_acc.detach().item())
      #del input
      #del label
      del output
      torch.cuda.empty_cache()

    mean_train_loss = sum(train_loss)/len(train_loss)
    mean_train_acc = sum(train_acc)/len(train_acc)
    
    #vbdata = next(iter(val_dataloader))
    with torch.no_grad():
      model.eval()
      for vbdata in tqdm(val_dataloader):
      #for i in range(1): # try to remove enumerate and bi to check if tqdm is working

          input = vbdata['input_image']
          label = vbdata['segmentation_image']
          input = input.to(device = config.device, dtype = torch.float16)
          label = label.to(device = config.device, dtype = torch.float16)

          with torch.cuda.amp.autocast(enabled=True):
            output = model(input)
            v_acc = DiceScore(output, label)
          
          scaler.scale(v_acc)
          val_acc.append(v_acc.detach().item())

          del output
          torch.cuda.empty_cache()
      
    mean_val_acc = sum(val_acc)/len(val_acc)

    print(f"training loss : {(mean_train_loss):.4f}, training accuracy : {(mean_train_acc):.4f}")
    print(f"validation accuracy : {(mean_val_acc):.4f}")

    #scheduler.step(mean_val_acc)
    scheduler.step()

    save_metrics(epoch, mean_train_loss,mean_train_acc, mean_val_acc)

    to_save = True if (epoch < 40 and epoch % 5==1) or (epoch >40 and epoch %10==9) else False
    if to_save is True:
      state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        'scheduler_dict': scheduler.state_dict(),
        'scaler_dict': scaler.state_dict(),
        'train_loss' : mean_train_loss,
        'val_acc' : mean_val_acc
              }

      ckpt_name = f'model_ckpt_{epoch}.pt'
      savepath = os.path.join(config.MODEL_SAVE_PATH,ckpt_name)
      torch.save(state,savepath)

    end = time.time()
    minutes, seconds = divmod(end-start, 60)     # try to improve time formatiting
    minutes = int(minutes)
    seconds = int(seconds)
    print(f"Epoch {epoch} ran {minutes}::{seconds}")


def resume_training_from_checkpoint(path):

  checkpoint = torch.load(path)

  ckpt_model = nnUNetModel().to(config.device)
  ckpt_optimizer = adw_optimizer
  ckpt_scheduler = scheduler

  ckpt_model.load_state_dict(checkpoint['state_dict'])
  ckpt_optimizer.load_state_dict(checkpoint['optimizer_dict'])
  ckpt_scheduler.load_state_dict(checkpoint['scheduler_dict'])
  ckpt_scaler.load_state_dict(checkpoint['scaler_dict'])
  epoch = checkpoint['epoch']

  run(start_epoch = epoch, total_epochs = config.NUM_EPCOHS, model = ckpt_model, train_dataloader = train_dataloader, val_dataloader = validation_dataloader, optimizer = ckpt_optimizer, scheduler = ckpt_scheduler)

RESUME_TRAINING = False
#RESUME_FROM_CHECKPOINT = '' if RESUME_TRAINING is True else None


if RESUME_TRAINING is True:
  files_list = glob.glob(config.MODEL_SAVE_PATH + '/*')
  ckpt_path = max(files_list, key=os.path.getctime)
  #ckpt_path = '/content/drive/MyDrive/AutomaticBrainTumorSegmentation/saved_models/nnUNet/experiment_2/model_ckpt_1.pt'
  resume_training_from_checkpoint(ckpt_path)
  
else:
  print(adw_optimizer)
  run(start_epoch = 1, total_epochs = config.NUM_EPOCHS, model = model, train_dataloader = train_dataloader, val_dataloader = validation_dataloader, optimizer = adw_optimizer, scheduler = scheduler)