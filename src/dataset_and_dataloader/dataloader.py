from torch.utils.data import DataLoader

from src.dataset_and_dataloader.dataset import LeafSegmentationDataset


def dataloaders(train_dataset: LeafSegmentationDataset, test_dataset: LeafSegmentationDataset, batch_size: int) -> tuple[DataLoader, DataLoader]:
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_dataloader, test_dataloader
