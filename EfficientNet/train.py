from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
import os
from efficientnet.model import EfficientNet
import numpy as np
import csv

# some parameters
use_gpu = torch.cuda.is_available()
print("torch.cuda.device_count()",torch.cuda.device_count())

# The train/test/val image root dir
data_dir = '/home/guoming/Dataset/Food_meituan/scaledata'
batch_size = 32
lr = 0.000001
momentum = 0.9
num_epochs = 100
input_size = 224
class_num = 1000
net_name = 'efficientnet-b7'


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
 
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
def loaddata(data_dir, batch_size, set_name, shuffle):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x), data_transforms[x]) for x in [set_name]}
    # num_workers=0 if CPU else =1
    dataset_loaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                      batch_size=batch_size,
                                                      shuffle=shuffle, num_workers=1) for x in [set_name]}
    data_set_sizes = len(image_datasets[set_name])
    return dataset_loaders, data_set_sizes


def train_model(model_ft, criterion, optimizer, lr_scheduler, num_epochs=300):
    train_loss = []
    since = time.time()
    best_model_wts = model_ft.state_dict()
    best_acc = 0.0
    model_ft.train(True)
    for epoch in range(num_epochs):
        dset_loaders, dset_sizes = loaddata(data_dir=data_dir, batch_size=batch_size, set_name='train', shuffle=True)
        print('Data Size', dset_sizes)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        optimizer = lr_scheduler(optimizer, epoch,lr)

        running_loss = 0.0
        running_corrects = 0
        count = 0

        for data in dset_loaders['train']:
            inputs, labels,path = data
            labels = torch.squeeze(labels.type(torch.LongTensor))
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            outputs = model_ft(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs.data, 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count += 1
            if count % 30 == 1 or outputs.size()[0] < batch_size:
                print('Epoch:{}: count:{} loss:{:.3f}'.format(epoch,count, loss.item()))
                train_loss.append(loss.item())

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dset_sizes
        epoch_acc = running_corrects.double() / dset_sizes

        print('Loss: {:.4f} Acc: {:.4f}'.format(
            epoch_loss, epoch_acc))

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model_ft.state_dict()
        if epoch_acc > 0.999:
            break
        if epoch%5==0:
            # save best model
            save_dir = data_dir + '/model'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            model_ft.load_state_dict(best_model_wts)
            model_out_path = save_dir + "/" + net_name +"%04d"%epoch+ '.pth'
            torch.save(model_ft, model_out_path)


    # save best model
    save_dir = data_dir + '/model'
    model_ft.load_state_dict(best_model_wts)
    model_out_path = save_dir + "/" + net_name + '.pth'
    torch.save(model_ft, model_out_path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return train_loss, best_model_wts


def val_model(model, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    cont = 0
    outPre = []
    outLabel = []
    dset_loaders, dset_sizes = loaddata(data_dir=data_dir, batch_size=1, set_name='val', shuffle=False)
    for data in dset_loaders['val']:
        inputs, labels = data
        labels = torch.squeeze(labels.type(torch.LongTensor))
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        outputs = model(inputs)
        output_np=outputs.data.cpu().numpy()[0]
        print(np.shape(output_np))
        output_idx=np.argsort(output_np)
        idx = output_idx[::-1]
        top_k_idx = list(idx[:3])
        if labels.data.item() in top_k_idx:
            running_corrects+=1
    print('Acc: {:.4f}'.format(float(running_corrects)/float(dset_sizes)))
        
    #     _, preds = torch.max(outputs.data, 1)
    #     loss = criterion(outputs, labels)
    #     if cont == 0:
    #         outPre = outputs.data.cpu()
    #         outLabel = labels.data.cpu()
    #     else:
    #         outPre = torch.cat((outPre, outputs.data.cpu()), 0)
    #         outLabel = torch.cat((outLabel, labels.data.cpu()), 0)
    #     running_loss += loss.item() * inputs.size(0)
    #     running_corrects += torch.sum(preds == labels.data)
    #     cont += 1
    # print('Loss: {:.4f} Acc: {:.4f}'.format(running_loss / dset_sizes,
    #                                         running_corrects.double() / dset_sizes))


def exp_lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=10):
    """Decay learning rate by a f#            model_out_path ="./model/W_epoch_{}.pth".format(epoch)
#            torch.save(model_W, model_out_path) actor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.8**(epoch // lr_decay_epoch))
    print('LR is set to {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


# train
pth_map = {
    'efficientnet-b0': 'efficientnet-b0-355c32eb.pth',
    'efficientnet-b1': 'efficientnet-b1-f1951068.pth',
    'efficientnet-b2': 'efficientnet-b2-8bb594d6.pth',
    'efficientnet-b3': 'efficientnet-b3-5fb5a3c3.pth',
    'efficientnet-b4': 'efficientnet-b4-6ed6700e.pth',
    'efficientnet-b5': 'efficientnet-b5-b6417697.pth',
    'efficientnet-b6': 'efficientnet-b6-c76e70fd.pth',
    'efficientnet-b7': 'efficientnet-b7-dcc49843.pth',
}
# automatically download model
# model_ft = EfficientNet.from_pretrained('efficientnet-b0')
# load local Network
model_ft = EfficientNet.from_name(net_name)

net_weight = 'pre_trained/' + pth_map[net_name]
state_dict = torch.load(net_weight)
model_ft=torch.load('/home/guoming/Dataset/Food_meituan/scaledata/model/efficientnet-b70020.pth')

# modify the last layer to fit class num
num_ftrs = model_ft._fc.in_features
model_ft._fc = nn.Linear(num_ftrs, class_num)

criterion = nn.CrossEntropyLoss()
if use_gpu:
    model_ft = model_ft.cuda()
    criterion = criterion.cuda()

optimizer = optim.SGD((model_ft.parameters()), lr=lr,
                      momentum=momentum, weight_decay=0.0004)

train_loss, best_model_wts = train_model(model_ft, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs)

val_model(model_ft,criterion)