import torch
import torchvision
from torch import nn
from torchvision import transforms,models,datasets
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import time
import copy
import json
import imageio
import warnings
import random
import sys
from PIL import Image

# 加载数据
data_dir='./lanslide_data/huapo_data'
train_dir=data_dir+'/train'
valid_dir=data_dir+'/valid'
data_transforms={
    'train':transforms.Compose([
        transforms.RandomRotation(45),#随机旋转，-45到45度之间随机选
        transforms.CenterCrop(224),#从中心开始裁剪
        transforms.RandomHorizontalFlip(p=0.5),#随机水平翻转，选择一个概率
        transforms.RandomVerticalFlip(p=0.5),#随机垂直翻转
        transforms.ColorJitter(brightness=0.2,contrast=0.1,saturation=0.1,hue=0.1),#参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
        transforms.RandomGrayscale(p=0.025),#概率转换为灰度率，3通道就是R=G=B
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    'valid':transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
}

batch_size=4
image_datasets={x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ['train','valid']}
# print(image_datasets['train'])
dataloaders={x:torch.utils.data.DataLoader(image_datasets[x],batch_size=batch_size,shuffle=True) for x in ['train','valid']}
# print(dataloaders)
data_sizes={x:len(image_datasets[x]) for x in ['train','valid']}
# print(data_sizes)
class_names=image_datasets['train'].classes
# print(class_names)
# print(image_datasets)
# print(dataloaders)
# print(data_sizes)
end_time=time.time()
model_name='resnet'
#是否用人家训练好的特征来做
feature_extract=True

#是否用GPU训练
train_on_gpu=torch.cuda.is_available()
if not train_on_gpu:
    print('cuda is not avaliable,training on cpu')
else:
    print('cuda is not avaliable,training on GPU')

device=torch.device("cuda:0"if torch.cuda.is_available() else "cpu")


def set_parameter_requires_grad(model,feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad=False
model_ft=models.resnet152()
def initialize_model(model_name,num_classes,feature_extract,use_pretrained=True):
    #选择合适的模型，不同的初始化方法稍微有点区别
    model_ft=None
    input_size=0
    if model_name=="resnet":
        """resnet152"""
        model_ft=models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft,feature_extract)
        num_ftrs=model_ft.fc.in_features
        model_ft.fc=nn.Sequential(nn.Linear(num_ftrs,2),nn.LogSoftmax(dim=1))
        input_size=224
    elif model_name=="vgg":
        model_ft=models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft,feature_extract)
        num_ftrs=model_ft.classifier[6].in_features
        model_ft.classifier[6]=nn.Linear(num_ftrs,num_classes)
        input_size=224
    else:
        print("invalid model name,exiting")
        exit()
    return model_ft,input_size
# 设置哪些层需要训练
model_ft,input_size=initialize_model(model_name,2,feature_extract,use_pretrained=True)
#使用GPU计算
model_ft=model_ft.to(device)
#模型保存
filename='checkpoint.pth'
# filename='seriouscheckpoint.pth'
#是否训练所有层
params_to_update=model_ft.parameters()
# print("params to learn")
if feature_extract:
    params_to_update=[]
    for name,param in model_ft.named_parameters():
        if param.requires_grad==True:
            params_to_update.append(param)
            # print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad==True:
            params_to_update.append(param)
            # print("\t",name)
#优化器设置
optimizer_ft=optim.Adam(params_to_update,lr = 0.01)
scheduler=optim.lr_scheduler.StepLR(optimizer_ft,step_size=7,gamma=0.1)
#最后一层已经有Logsoftmax()了，所以不能nn,crossentropyloss()来计算了
criterion=nn.NLLLoss()

#训练模块
def train_model(model,dataloaders,criterion,optimizer,num_epochs=25,is_inception=False,filename=filename):
    since=time.time()
    best_acc=0
    """
    checkpoint=torch.load(filename)
    best_acc=checpoint['best_acc']
    model.load_state_dict(checkpoint['optimizer'])
    model.class_to_idx=checkpoint['mapping']
    """
    #使用GPU
    # model.to(device)
    val_acc_history=[]
    train_acc_history=[]
    train_losses=[]
    valid_losses=[]
    LRs=[optimizer.param_groups[0]['lr']]
    best_model_wts=copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('Epoch{}/{}'.format(epoch,num_epochs-1))
        print(' * '*10)

        #训练和验证阶段
        for phase in ['train','valid']:
            if phase=='train':
                model.train()#训练
            else:
                model.eval()#验证

            running_loss=0.0
            running_corrects=0

            #把数据都取个遍
            for inputs,labels in dataloaders[phase]:
                inputs=inputs.to(device)
                labels=labels.to(device)

                #清零
                optimizer.zero_grad()

                #只有训练的时候计算和更新梯度
                with torch.set_grad_enabled(phase=='train'):
                    if is_inception and phase=='train':
                        outputs,aux_outputs=model(inputs)
                        loss1=criterion(outputs,labels)
                        loss2=criterion(aux_outputs,labels)
                        loss=loss1+loss2
                    else:
                        outputs=model(inputs)
                        loss=criterion(outputs,labels)
                        _,preds=torch.max(outputs,1)
                    #训练更新权重
                    if phase=='train':
                        loss.backward()
                        optimizer.step()
                #计算损失
                running_loss+=loss.item()*inputs.size(0)
                running_corrects+=torch.sum(preds==labels.data)
            epoch_loss=running_loss/len(dataloaders[phase].dataset)
            epoch_acc=running_corrects.double()/len(dataloaders[phase].dataset)


            time_elapsed=time.time()-since
            print('time elapsed  {:.0f}m {:.0f}s'.format(time_elapsed//60,time_elapsed%60))
            print('{}loss: {:.4f} Acc: {:.4f}'.format(phase,epoch_loss,epoch_acc))

            #得到最好的那次模型
            if phase=='valid' and epoch_acc>best_acc:
                best_acc=epoch_acc
                best_model_wts=copy.deepcopy(model.state_dict())
                state={
                    'state_dict':model.state_dict(),
                    'best_acc':best_acc,
                    'optimizer':optimizer.state_dict()
                }
                torch.save(state,filename)

            if phase=='valid':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
                scheduler.step(epoch_loss)
            if phase=='train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)
        print('optimizer learning rate: {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])

    time_elapsed=time.time()-since
    print('training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60,time_elapsed%60))
    print('best val Acc:{:4f}'.format(best_acc))

    #训练完之后把最好的一次当作模型最终的结果
    model.load_state_dict(best_model_wts)
    return model,val_acc_history,train_acc_history,valid_losses,train_losses,LRs



#开始训练
model_ft,val_acc_history,train_acc_history,valid_losses,train_losses,LRs=train_model(model_ft,dataloaders,criterion,optimizer_ft,num_epochs=5,is_inception=False,filename=filename)





