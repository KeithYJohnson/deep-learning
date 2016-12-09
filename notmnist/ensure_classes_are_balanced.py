import os
from data_locations import *
from pprint import pprint
from scipy import mean

def ensure_classes_are_balanced():
    print('ensuring classes are balanced')
    check_folder(train_folders)
    check_folder(test_folders)

def check_folder(folders):
    print('checking that we have roughly the same amount of samples from each class in {}'.format(folders))
    folder_counts = {}
    for folder in folders:
        folder_counts[folder] = len(os.listdir(folder))

    average = mean(list(folder_counts.values()))
    for folder, num_items in folder_counts.items():
        if abs(num_items - average) > average * .1:
            print("{} number of items is significantly different from the mean".format(folder))



    pprint(folder_counts)

ensure_classes_are_balanced()
