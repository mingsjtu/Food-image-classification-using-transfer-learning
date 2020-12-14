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
data_dir = '/home/guoming/Dataset/Food_meituan/scaledata'
batch_size = 32
lr = 0.01
momentum = 0.9
num_epochs = 60
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



def test_model(model,output_csv):
    model.eval()
    f_csv = open(output_csv,'w',encoding='utf-8',newline="")
    csv_writer = csv.writer(f_csv)
    csv_writer.writerow(["id","predicted"])
    dset_loaders, dset_sizes = loaddata(data_dir=data_dir, batch_size=16, set_name='test_new', shuffle=False)
    num=0
    output_all=np.zeros((50000,1000))
    fix_output_np=np.zeros((1000))

    label_np=np.load('label_np.npy')

    for data in dset_loaders['test_new']:
        inputs, labels,allpath = data
        labels = torch.squeeze(labels.type(torch.LongTensor))
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        outputs = model(inputs)
        print(allpath)
        output_np=np.zeros((1000))
        for j in range(16):
            output_np+=outputs.data.cpu().numpy()[j]
        path=os.path.split(allpath[0])[1]
        print(path)
        for i in range(1000):
            fix_output_np[int(label_np[i])]=output_np[i]
        path_id=int(path.split('_')[1].split('.')[0])
        output_all[path_id]=fix_output_np

        output_idx=np.argsort(fix_output_np)
        idx = output_idx[::-1]
        top_k_idx = list(idx[:3])
        
        csv_writer.writerow(["test_%04d.jpg"%path_id,str(top_k_idx[0])+' '+str(top_k_idx[1])+' '+str(top_k_idx[2])])
        print(["test_%04d.jpg"%path_id,str(top_k_idx[0])+' '+str(top_k_idx[1])+' '+str(top_k_idx[2])])
        if num%100==1:
            print("Processing %4d"%num)
        num+=1



def exp_lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=10):
    """Decay learning rate by a f#            model_out_path ="./model/W_epoch_{}.pth".format(epoch)
#            torch.save(model_W, model_out_path) actor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.8**(epoch // lr_decay_epoch))
    print('LR is set to {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


# main function: test model
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

# The path of the trained model
net_weight = '/home/guoming/Dataset/Food_meituan/scaledata/model/efficientnet-b70020.pth'
model_ft=torch.load(net_weight)
print('-' * 10)
print('Test Accuracy:')
criterion = nn.CrossEntropyLoss().cuda()

# Save to the result csv file
test_model(model_ft, "gm_eff_val16.csv")
