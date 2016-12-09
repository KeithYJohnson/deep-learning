import os
from random import randint
from PIL import Image

train_folders = ['/Users/keithjohnson/datasets/notMNIST_large/A', '/Users/keithjohnson/datasets/notMNIST_large/B', '/Users/keithjohnson/datasets/notMNIST_large/C', '/Users/keithjohnson/datasets/notMNIST_large/D', '/Users/keithjohnson/datasets/notMNIST_large/E', '/Users/keithjohnson/datasets/notMNIST_large/F', '/Users/keithjohnson/datasets/notMNIST_large/G', '/Users/keithjohnson/datasets/notMNIST_large/H', '/Users/keithjohnson/datasets/notMNIST_large/I', '/Users/keithjohnson/datasets/notMNIST_large/J']
test_folders  = ['/Users/keithjohnson/datasets/notMNIST_small/A', '/Users/keithjohnson/datasets/notMNIST_small/B', '/Users/keithjohnson/datasets/notMNIST_small/C', '/Users/keithjohnson/datasets/notMNIST_small/D', '/Users/keithjohnson/datasets/notMNIST_small/E', '/Users/keithjohnson/datasets/notMNIST_small/F', '/Users/keithjohnson/datasets/notMNIST_small/G', '/Users/keithjohnson/datasets/notMNIST_small/H', '/Users/keithjohnson/datasets/notMNIST_small/I', '/Users/keithjohnson/datasets/notMNIST_small/J']

def display_samples():
    print("running displaying samples")
    for folder in train_folders + test_folders:
        files = os.listdir(folder)
        file = files[randint(0,len(files))]
        print('Randomly picked {} from {} to show'.format(file, folder))
        img = Image.open("{}/{}".format(folder,file))
        img.show()

display_samples()
