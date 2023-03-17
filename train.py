# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 17:38:02 2018
@author: Tao Lin

Implementation of the W-Net unsupervised image segmentation architecture
"""

import argparse
import torch.nn as nn
import numpy as np
import time
import datetime
import torch
from torch.utils.data import Dataset
import rasterio
import torch.nn.functional as F
from torchvision import datasets, transforms
from utils.org_soft_n_cut_loss import batch_soft_n_cut_loss
from utils.soft_n_cut_loss import soft_n_cut_loss

import WNet
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation with WNet')
parser.add_argument('--name', metavar='name', default=str(datetime.datetime.now().strftime('%Y%m%d%H%M%S')), type=str,
                    help='Name of model')
parser.add_argument('--in_Chans', metavar='C', default=3, type=int, 
                    help='number of input channels')
parser.add_argument('--squeeze', metavar='K', default=4, type=int, 
                    help='Depth of squeeze layer')
parser.add_argument('--out_Chans', metavar='O', default=3, type=int, 
                    help='Output Channels')
parser.add_argument('--epochs', metavar='e', default=100, type=int, 
                    help='epochs')
parser.add_argument('--input_folder', metavar='f', default=None, type=str, 
                    help='Folder of input images')
parser.add_argument('--output_folder', metavar='of', default=None, type=str, 
                    help='folder of output images')

softmax = nn.Softmax2d()
criterionIdt = torch.nn.MSELoss()

class CustomImageDataset(Dataset):
    def __init__(self, img_path='../Data/Geotiff-large-area/grid.tif', label_path="../Data/mounds_centers.png", size = 256, transform=None, target_transform=None, stride = 50):
        img = rasterio.open(img_path)
        img = img.read()[0]
        res_dic = {}
        res_dic["grid"] = img #(img[:,:] - img.min())/(img.max() - img.min())
        res_dic["gradient_0"] = np.gradient(res_dic["grid"][:,:],axis=0)[:,:]
        res_dic["gradient_1"] = np.gradient(res_dic["grid"][:,:],axis=1)[:,:]
        res_dic["slope"] = (np.arctan(np.sqrt(res_dic["gradient_0"]**2 + res_dic["gradient_1"]**2)))
        img = np.stack([res_dic["gradient_0"], res_dic["gradient_1"], res_dic["slope"]], axis=-1)
        self.img = torch.Tensor(img)
        label = plt.imread(label_path)
        label = label.sum(axis = -1 ) > 0
        self.img_label = F.one_hot(torch.Tensor(label[:,:]).long())
        self.transform = transform
        self.target_transform = target_transform
        self.size = size
        self.stride = stride

    def __len__(self):
        height, length = self.img.shape[:2]
        height -= self.size
        height //= self.stride
        length -= self.size
        length //= self.stride
        return height*length

    def __getitem__(self, idx):
        height, length = self.img.shape[:2]
        height -= self.size
        height //= self.stride
        length -= self.size
        length //= self.stride
        idx1 = idx // height
        idx1 *= self.stride
        idx0 = idx % height
        idx0 *= self.stride
        image = self.img[idx0:(idx0+self.size), idx1:(idx1+self.size)]
        label = self.img_label[idx0:(idx0+self.size), idx1:(idx1+self.size)]
        if self.transform:
            try:
                image = self.transform(image)
            except:
                image = image.numpy()
                image = self.transform(image)
        if self.target_transform:
            try:
                label = self.target_transform(label)
            except:
                label = label.numpy()
                label = self.target_transform(label)
        print(image.shape)
        print(label.shape)
        return image.permute(2, 0, 1), label.permute(2, 0, 1)

def train_op(model, optimizer, input, k, img_size, psi=0.5):
    enc = model(input, returns='enc')
    d = enc.clone().detach()
    n_cut_loss=soft_n_cut_loss(input,  softmax(enc),  img_size)
    n_cut_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    dec = model(input, returns='dec')
    rec_loss=reconstruction_loss(input, dec)
    rec_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return (model, n_cut_loss, rec_loss)

def reconstruction_loss(x, x_prime):
    rec_loss = criterionIdt(x_prime, x)
    return rec_loss

def test():
    wnet=WNet.WNet(4)
    synthetic_data=torch.rand((1, 3, 128, 128))
    optimizer=torch.optim.SGD(wnet.parameters(), 0.001) #.cuda()
    train_op(wnet, optimizer, synthetic_data)

def show_image(image):
    img = image.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    plt.show()

def main():
    # Load the arguments
    args, unknown = parser.parse_known_args()

    # Check if CUDA is available
    CUDA = torch.cuda.is_available()

    # Create empty lists for average N_cut losses and reconstruction losses
    n_cut_losses_avg = []
    rec_losses_avg = []

    # Squeeze k
    k = args.squeeze
    img_size = (224, 224)
    wnet = WNet.WNet(k)
    if(CUDA):
        wnet = wnet.cuda()
    learning_rate = 0.003
    optimizer = torch.optim.SGD(wnet.parameters(), lr=learning_rate)

    transform = transforms.Compose([transforms.Resize(img_size),
                                transforms.ToTensor()])

    dataset = CustomImageDataset(img_path=args.input_folder, label_path=args.output_folder, transform=transform)

    # Train 1 image set batch size=1 and set shuffle to False
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

    # Run for every epoch
    for epoch in range(args.epochs):

        # At 1000 epochs divide SGD learning rate by 10
        if (epoch > 0 and epoch % 1000 == 0):
            learning_rate = learning_rate/10
            optimizer = torch.optim.SGD(wnet.parameters(), lr=learning_rate)

        # Print out every epoch:
        print("Epoch = " + str(epoch))

        # Create empty lists for N_cut losses and reconstruction losses
        n_cut_losses = []
        rec_losses = []
        start_time = time.time()

        for (idx, batch) in enumerate(dataloader):
            # Train 1 image idx > 1
            # if(idx > 1): break

            # Train Wnet with CUDA if available
            if CUDA:
                batch[0] = batch[0].cuda()
            
            wnet, n_cut_loss, rec_loss = train_op(wnet, optimizer, batch[0], k, img_size)

            n_cut_losses.append(n_cut_loss.detach())
            rec_losses.append(rec_loss.detach())

        n_cut_losses_avg.append(torch.mean(torch.FloatTensor(n_cut_losses)))
        rec_losses_avg.append(torch.mean(torch.FloatTensor(rec_losses)))
        print("--- %s seconds ---" % (time.time() - start_time))


    images, labels = next(iter(dataloader))

    # Run wnet with cuda if enabled
    if CUDA:
        images = images.cuda()

    enc, dec = wnet(images)

    torch.save(wnet.state_dict(), "model_" + args.name)
    np.save("n_cut_losses_" + args.name, n_cut_losses_avg)
    np.save("rec_losses_" + args.name, rec_losses_avg)
    print("Done")

if __name__ == '__main__':
    main()


# python .\train.py --e 100 --input_folder="data/images/" --output_folder="/output/"