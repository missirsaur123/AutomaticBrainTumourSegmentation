import torch
import torch.nn as nn
import torch.nn.functional as F

def WeightedMultiClassCELoss(predict, target, device = 'cuda', reduce = 'mean'):
    """
    predict : Torch.Tensor(1,3,128,128,128)
    
    """
    assert predict.shape == target.shape, "predict & target size doesn't match"  

    bs, num_channels = target.shape[0], target.shape[1]

    predict = predict.contiguous().view(-1,num_channels)
    target = target.contiguous().view(-1,num_channels)
    
    #class_weights = torch.FloatTensor([50,40,60]).to(device = device)
    class_weights = torch.FloatTensor([2,5,15]).to(device = device)
  
    celoss = nn.BCEWithLogitsLoss(pos_weight = class_weights)
    
    weighted_loss = celoss(predict,target)
    #print(weighted_loss)
    return weighted_loss

def MultiClassCELoss(predict, target, device = 'cuda'):
    """
    predict : Torch.Tensor(1,3,128,128,128)
    
    """
    assert predict.shape == target.shape, "predict & target batch size don't match"  
    K = target.shape[1]
    bceloss = nn.BCELoss()
    loss = bceloss(predict,target)

    return loss
   

def MultiClassDiceLoss(predict,target,device = 'cuda', reduce = None, apply_sigmoid = True): 

    assert predict.shape == target.shape, f"predict{predict.shape} & target{target.shape} batch size don't match"  

    smooth = 1
    p = 1
    epsilon = 1e-3
    
    bs, num_channels = target.shape[0], target.shape[1]

    if apply_sigmoid :
        predict = torch.sigmoid(predict)

    predict = predict.contiguous().view(bs,num_channels,-1)
    target = target.contiguous().view(bs,num_channels,-1)

    num = 2.0*torch.mul(predict, target).sum(2).sum(0) + smooth
     
    den = torch.add(predict,target).sum(2).sum(0) + smooth

    loss = 1 - (torch.sum(num)/torch.sum(den))

    return loss

def GeneralizedDiceLoss(predict,target,device = 'cuda', reduce = None, apply_sigmoid = True):

    assert predict.shape == target.shape, f"predict{predict.shape} & target{target.shape} shape doesn't match"  
    smooth = 1
    p = 1
    epsilon = 1e-3
    
    bs, num_channels = target.shape[0], target.shape[1]

    we = [target[:,i,:,:,:].sum().item() for i in range(3)]
    
    class_weights = [1/(w**2 + epsilon)  for w in we]
    #class_weights = [10**3/(w**2 + epsilon)  for w in we]
    #print(class_weights)
    
    #class_weights_array = torch.FloatTensor(class_weights).to(device)
    class_weights_array = torch.FloatTensor(class_weights).to(device).clamp(min = 1e-8, max = 1e-4)    
    #print(class_weights_array)

    if apply_sigmoid is True:
        predict = torch.sigmoid(predict)

    predict = predict.contiguous().view(bs,num_channels,-1)
    target = target.contiguous().view(bs,num_channels,-1)

    num = 2.0*torch.mul(predict, target).sum(2).sum(0) + smooth
    #print(num.shape) 
    num = torch.mul(num,class_weights_array)
    #den = torch.sum(new_predict.pow(p)),torch.sum(target.pow(p)) + smooth
    den = torch.add(predict,target).sum(2).sum(0) + smooth
    #print(den.shape)
    den = torch.mul(den,class_weights_array) 
    #print(num,den)

    loss = 1 - (torch.sum(num)/torch.sum(den))

    return loss

def BCE_plus_Dice_Loss(predict, target):

    dice_loss = MultiClassDiceLoss(predict, target)
    ce_loss = MultiClassCELoss(predict, target)

    total_loss = dice_loss + ce_loss

    return total_loss

def Weighted_CE_plus_Dice_Loss(predict,target,alpha = 1):

    weighted_ce_loss = WeightedMultiClassCELoss(predict,target)
    weighted_dice_loss = GeneralizedDiceLoss(predict,target, apply_sigmoid = True)

   
    #total_loss = weighted_ce_loss + weighted_dice_loss
    total_loss = alpha*weighted_ce_loss + weighted_dice_loss
    #print(f'CE Loss : {weighted_ce_loss}, Dice Loss : {weighted_dice_loss}')

    return total_loss