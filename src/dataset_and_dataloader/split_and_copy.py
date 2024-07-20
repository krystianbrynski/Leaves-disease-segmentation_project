import os
import shutil

from sklearn.model_selection import train_test_split


def split_data(images_dir, masks_dir, train_masks_dir, test_masks_dir, train_images_dir, test_images_dir):
    images_file = os.listdir(images_dir)  # all files in images dir
    masks_file = os.listdir(masks_dir)  # all files in masks dir

    os.makedirs(train_masks_dir, exist_ok=True)  # make new dir for masks train
    os.makedirs(test_masks_dir, exist_ok=True)  # make new dir for masks test
    os.makedirs(train_images_dir, exist_ok=True)  # make new dir for images train
    os.makedirs(test_images_dir, exist_ok=True)  # make new dir for images test+

    train_images, test_images, train_masks, test_masks = train_test_split(images_file, masks_file, test_size=0.1, random_state=42)

    for i in train_images:
        source_file = images_dir + '/' + i  # path to image file
        shutil.copy(source_file, train_images_dir)  # photo copy to new dir

    for i in test_images:
        source_file = images_dir + '/' + i  # path to image file
        shutil.copy(source_file, test_images_dir)  # photo copy to new dir

    for i in train_masks:
        source_file = masks_dir + '/' + i  # path to image file
        shutil.copy(source_file, train_masks_dir)  # photo copy to new dir

    for i in test_masks:
        source_file = masks_dir + '/' + i  # path to image file
        shutil.copy(source_file, test_masks_dir)  # photo copy to new dir
