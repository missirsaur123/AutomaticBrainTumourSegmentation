import torch
import torch.nn as nn

def DiceScore(output, target):

    #returns 3 outputs as dice scores of wt,
    assert output.shape == target.shape, "predict & target batch size don't match"  
    smooth = 1e-3

    K = target.shape[1]
    #predict = output.contiguous.view(K,-1)
    outupt = torch.sigmoid(output)
    
    num = 2. * torch.sum(torch.mul(output, target)).item() + smooth
    den = torch.sum(torch.add(output,target)).item() + smooth
    
    return (num/den)


def dice_coef(pred, target):
    
    K = target.shape[0]
    pred = pred.contiguous().view(K,-1)
    target = target.contiguous().view(K,-1)
    
    smooth = 1e-3

    num = 2. * torch.sum(torch.mul(pred, target)).item() + smooth
    den = torch.sum(torch.add(pred,target)).item() + smooth
    
    return (num/den)

def ClassDiceScore(predict, target):

    predict = torch.sigmoid(predict)
    class_dice = []

    wt_dice = dice_coef(predict[:,0:,:,:,:], target[:,0:,:,:,:])
    ed_dice = dice_coef(predict[:,1:,:,:,:], target[:,1:,:,:,:])
    et_dice = dice_coef(predict[:,2,:,:,:], target[:,2,:,:,:])
    
    #print(f'whole tumour : {class_dice[0]:.4f}, tumour core : {class_dice[1]:.4f}, enhancing tumour : {class_dice[2]:.4f}')

    return (wt_dice, ed_dice, et_dice)
    
def pixel_count(output, target):

    sm = nn.Softmax(dim = 1)
    output = sm(output)
    output = output > 0.5
    
    print('\n')
    print('Classwise output pixel count')
    print(torch.unique(output[:,0,:,:,:],return_counts = True))
    print(torch.unique(output[:,1,:,:,:],return_counts = True))
    print(torch.unique(output[:,2,:,:,:],return_counts = True))
    print('\n')
    print('Classwise target pixel count')
    print(torch.unique(target[:,0,:,:,:],return_counts = True))
    print(torch.unique(target[:,1,:,:,:],return_counts = True))
    print(torch.unique(target[:,2,:,:,:],return_counts = True))
    

def accuracy(output,target):
    
    pred = (output > 0.5).int()
    corrects = (pred == target).int().sum()
    total = target.numel()
    print(corrects,total)
    acc = (corrects /total )*100
    return acc


def class_accuracy(output,target):
    
    class_acc = []
    class_acc.append(accuracy(normalized_pred[:,0,:,:,:],target[:,0,:,:,:]))
    class_acc.append(accuracy(normalized_pred[:,1,:,:,:],target[:,1,:,:,:]))
    class_acc.append(accuracy(normalized_pred[:,2,:,:,:],target[:,2,:,:,:]))
    class_acc.append(accuracy(normalized_pred[:,3,:,:,:],target[:,3,:,:,:]))

    print(f"Class Accuracies:- \nbg = {class_acc[0]:.3f}, ncr = {class_acc[1]:.3f}, ed = {class_acc[2]:.3f}, et = {class_acc[3]:.3f}")
    pred = output.argmax(dim = 1)
    actual = target.argmax(dim = 1)
    corrects = (pred == actual).int().sum()
    total = actual.numel()
    acc = (corrects /total )*100
    print(f'Segmentation Accuracy = {acc:.2f}')

    return class_acc 