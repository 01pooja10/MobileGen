import torch
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def plot_imgs(image_set):
    plt.figure(32,32)
    plt.imshow(torch.cat([torch.cat([img for img in image_set.cpu().detach()],dim=-1),],dim=-2).permute(1,2,0).cpu())
    plt.show()
    
def save_imgs(image_set,save_pth):
    grid_im = torchvision.utils.make_grid(image_set,)
    arr = grid_im.permute(1,2,0).cpu().numpy()
    img = Image.fromarray(arr)
    img.save(save_pth)
    

def get_data(dpath,batch_size):
    
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(32),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,),(0.5,))
    ])
    
    #dataset = torchvision.datasets.ImageFolder(dpath, transform=transforms)
    dataset = torchvision.datasets.CIFAR10(dpath,download=False,transform=transforms)
    
    to_filter = list(range(0,len(dataset),10))
    new_data = torch.utils.data.Subset(dataset,to_filter)
    
    dataloader = DataLoader(new_data, batch_size = batch_size, shuffle = True)
    
    
    #print(len(dataloader))
    
    return dataloader
    
    
