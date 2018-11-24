import torch
import torchvision
import torch
import sys
from transform import transform_training, transform_testing
import torchvision.datasets as datasets
import config as cf
import os
import numpy as np
data_path = './data'
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
def splitTrainAndValidate(mydset):
    num_train = len(mydset)
    indices = list(range(num_train))
    split = int(num_train *0.3)
    # Random, non-contiguous split
    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))
    # Contiguous split
    # train_idx, validation_idx = indices[split:], indices[:split]

    ## define our samplers -- we use a SubsetRandomSampler because it will return
    ## a random subset of the split defined by the given indices without replaf
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    train_loader = torch.utils.data.DataLoader(mydset, batch_size=cf.batch_size, num_workers=4, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(mydset, batch_size=cf.batch_size, num_workers=4, sampler=validation_sampler)
    return train_loader,validation_loader
def dataset(dataset_name):

    if (dataset_name == 'cifar10'):
        print("| Preparing CIFAR-10 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_training())
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_testing())
        outputs = 10
        inputs = 3
    
    elif (dataset_name == 'cifar100'):
        print("| Preparing CIFAR-100 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_training())
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_testing())
        outputs = 100
        inputs = 3
    
    elif (dataset_name == 'mnist'):
        print("| Preparing MNIST dataset...")
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_training())
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_testing())
        outputs = 10
        inputs = 1
    elif(dataset_name  == 'dog-breed'):
        print("| Preparing dog-breed dataset...")
        trainset = datasets.ImageFolder(os.path.join(data_path, 'train'),transform=transform_training())
        print(trainset)
        testset = datasets.ImageFolder(os.path.join(data_path, 'test'),transform=transform_testing())                  
        outputs = 120
        inputs = 3
    elif (dataset_name == 'fashionmnist'):
        print("| Preparing FASHIONMNIST dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_training())
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform_testing())
        outputs = 10
        inputs = 1
    elif (dataset_name == 'stl10'):
        print("| Preparing STL10 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.STL10(root='./data',  split='train', download=True, transform=transform_training())
        testset = torchvision.datasets.STL10(root='./data',  split='test', download=False, transform=transform_testing())
        outputs = 10
        inputs = 3
    trainloader, validateloader = splitTrainAndValidate(trainset)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=cf.batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=cf.batch_size, shuffle=False, num_workers=4)
    
    return  trainloader, validateloader,testloader, outputs, inputs

