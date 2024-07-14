from typing import Any, Dict, Optional

import os
import shutil
import tarfile
from .components.dataset_manager import DatasetManager
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Subset


class IP102DataModule(LightningDataModule):
    """Data module for the IP102 dataset.

    This LightningDataModule provides functionality to prepare, load, and transform the IP102 dataset
    for training, validation, and testing purposes. It handles dataset preparation, including initializing
    and managing the dataset, creating necessary folders, loading class labels, creating subfolders for each class,
    and moving files to their respective subfolders.

    Args:
        data_dir (str): The directory path to the IP102 dataset. Default is "/home/adhi/large-scale-pest-classification/data/ip102_v1.1/".
        batch_size (int): The batch size to use for data loaders. Default is 64.
        num_workers (int): The number of workers for data loading. Default is 0.
        pin_memory (bool): If True, pin memory for faster data transfer to the GPU. Default is False.

    Attributes:
        num_classes (int): The number of classes in the IP102 dataset.

    Methods:
        prepare_data(): Prepares the dataset by initializing and managing the dataset, creating necessary folders,
            loading class labels, creating subfolders for each class, and moving files to their respective subfolders.
        setup(): Loads and splits the datasets if they are not already loaded.
        train_dataloader(): Returns a DataLoader for the training dataset.
        val_dataloader(): Returns a DataLoader for the validation dataset.
        test_dataloader(): Returns a DataLoader for the testing dataset.
        teardown(stage: Optional[str] = None): Cleans up after fit or test.
        state_dict(): Returns extra things to save to checkpoint.
        load_state_dict(state_dict: Dict[str, Any]): Performs actions when loading checkpoint.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        train_dir: str = "data/ip102_v1.1/images/train",
        val_dir: str = "data/ip102_v1.1/images/val",
        test_dir: str = "data/ip102_v1.1/images/test",
        batch_size: int = 64,
        num_workers: int = 16,
        pin_memory: bool = False,
        subset_fraction: float = 1.0,  # 添加新参数
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_dir = data_dir
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.subset_fraction = subset_fraction  # 保存新参数

        # data augmentation
        self.augmentation = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(224, scale=(0.7, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])                           
        ])

        # data transformations
        self.transforms = transforms.Compose([
            transforms.Resize(230),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return 102

    def prepare_data(self):
        """Prepares the dataset by initializing and managing the dataset, creating necessary folders,
        loading class labels, creating subfolders for each class, and moving files to their respective subfolders.

        Args:
            self: The instance of the class.

        Returns:
            None
        """

        # Directory path for the dataset
        dataset_path = self.data_dir

        # path for checking dataset
        check_path = f"{self.data_dir}images/train"

        # check if dataset already exist
        if not os.path.exists(check_path):

            # extract dataset
            tar_dir = os.path.dirname(dataset_path)
            tar_dir = tar_dir[:tar_dir.rfind('/') + 1]
            tar = tarfile.open(f"{tar_dir}ip102_v1.1.tar")
            tar.extractall(tar_dir)

            # copy class
            source_file = os.path.join(tar_dir, "classes.txt")
            destination_file = os.path.join(dataset_path, "classes.txt")
            shutil.copyfile(source_file, destination_file)

            # Initialize and manage the dataset
            dataset_manager = DatasetManager(dataset_path)

            # Create necessary folders
            dataset_manager.create_folders()

            # Load class labels
            classes_file = os.path.join(dataset_path, "classes.txt")
            dataset_manager.load_class_labels(classes_file)

            # Create subfolders for each class
            dataset_manager.create_class_subfolders()

            # Move files to respective subfolders
            dataset_manager.move_files("train.txt")
            dataset_manager.move_files("test.txt")
            dataset_manager.move_files("val.txt")


    def setup(self, stage=None):
        """Loads and splits the datasets if they are not already loaded.

        This method is called by Lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!

        Args:
            self: The instance of the class.

        Returns:
            None
        """

        # load and split datasets only if not loaded already
        # self.data_train = datasets.ImageFolder(self.train_dir, transform=self.augmentation)
        # self.data_val = datasets.ImageFolder(self.val_dir, transform=self.transforms)
        # self.data_test = datasets.ImageFolder(self.test_dir, transform=self.transforms)
        # 如果数据集已加载，避免重复加载
        if self.data_train is None:
            if self.subset_fraction < 1.0:
                # 如果设置了 subset_fraction，仅对训练数据集加载完整数据集并应用子集逻辑
                full_train_dataset = datasets.ImageFolder(self.train_dir, transform=self.augmentation)
                subset_size = int(len(full_train_dataset) * self.subset_fraction)
                indices = np.random.choice(len(full_train_dataset), subset_size, replace=False)
                self.data_train = Subset(full_train_dataset, indices)
                print(f"Loaded training subset: {len(self.data_train)} / {len(full_train_dataset)}")
            else:
                self.data_train = datasets.ImageFolder(self.train_dir, transform=self.augmentation)
                print(f"Loaded full training data: {len(self.data_train)}")

        if self.data_val is None:
            self.data_val = datasets.ImageFolder(self.val_dir, transform=self.transforms)
            print(f"Loaded validation data: {len(self.data_val)}")

        if self.data_test is None:
            self.data_test = datasets.ImageFolder(self.test_dir, transform=self.transforms)
            print(f"Loaded test data: {len(self.data_test)}")


    def train_dataloader(self):
        """Returns a DataLoader for the training dataset.

        Returns:
            DataLoader: The DataLoader object for the training dataset.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        """Returns a DataLoader for the validation dataset.

        Returns:
            DataLoader: The DataLoader object for the validation dataset.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        """Returns a DataLoader for the testing dataset.

        Returns:
            DataLoader: The DataLoader object for the testing dataset.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = IP102DataModule()
