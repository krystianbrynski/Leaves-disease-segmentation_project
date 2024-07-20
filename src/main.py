import click

import yaml

from dataset_and_dataloader import dataset, dataloader, split_and_copy


@click.command()
@click.option('--config', '-c', default='../config.yaml', help='Path to the configuration file')
def run_pipeline(config) -> None:
    with open(config, 'r') as file:
        config_data = yaml.safe_load(file)

        batch_size = config_data.get('batch_size')

        images_dir = config_data.get('images')  # load images dir
        masks_dir = config_data.get('masks')  # load masks dir

        train_masks_dir = config_data.get('train_masks')  # load dir for train masks
        test_masks_dir = config_data.get('test_masks')  # load dir for test masks
        train_images_dir = config_data.get('train_images')  # load dir for train images
        test_images_dir = config_data.get('test_images')  # load dir for test images

        split_and_copy.split_data(images_dir, masks_dir, train_masks_dir, test_masks_dir, train_images_dir, test_images_dir)  # functionality split data and copy to new dir

        train_dataset, test_dataset = dataset.datasets(train_images_dir, test_images_dir, train_masks_dir, test_masks_dir)

        train_dataloader, test_dataloader = dataloader.dataloaders(train_dataset, test_dataset, batch_size)





if __name__ == "__main__":
    run_pipeline()
