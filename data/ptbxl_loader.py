# PTB-XL Data Loader and Training Utilities
# For ECG classification with SpikeMamba

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.utils import shuffle
from natsort import natsorted


def normalize_batch_ts(batch):
    """Normalize a batch of time-series data.

    Args:
        batch (numpy.ndarray): A batch of input time-series in shape (N, T, C).

    Returns:
        numpy.ndarray: A batch of processed time-series, normalized for each channel of each sample.
    """
    # Calculate mean and std for each sample's each channel
    mean_values = batch.mean(axis=1, keepdims=True)  # Shape: (N, 1, C)
    std_values = batch.std(axis=1, keepdims=True)  # Shape: (N, 1, C)

    # Avoid division by zero by setting small std to 1
    std_values[std_values == 0] = 1.0

    # Perform standard normalization
    normalized_batch = (batch - mean_values) / std_values

    return normalized_batch


class PTBXLLoader(Dataset):
    """PTB-XL ECG Dataset Loader
    
    5-class ECG classification:
        0: Normal
        1: Myocardial Infarction (MI)
        2: ST/T Change (STTC)
        3: Conduction Disturbance (CD)
        4: Hypertrophy (HYP)
    """
    
    def __init__(self, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        a, b = 0.6, 0.8

        # list of IDs for training, val, and test sets
        self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(
            self.label_path, a, b
        )

        self.X, self.y = self.load_ptbxl(self.data_path, self.label_path, flag=flag)

        # pre_process
        self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]
        self.num_channels = self.X.shape[2]
        self.num_classes = len(np.unique(self.y))

    def load_train_val_test_list(self, label_path, a=0.6, b=0.8):
        """
        Loads IDs for training, validation, and test sets
        """
        data_list = np.load(label_path)
        no_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])
        mi_list = list(data_list[np.where(data_list[:, 0] == 1)][:, 1])
        sttc_list = list(data_list[np.where(data_list[:, 0] == 2)][:, 1])
        cd_list = list(data_list[np.where(data_list[:, 0] == 3)][:, 1])
        hyp_list = list(data_list[np.where(data_list[:, 0] == 4)][:, 1])

        train_ids = (
            no_list[: int(a * len(no_list))]
            + mi_list[: int(a * len(mi_list))]
            + sttc_list[: int(a * len(sttc_list))]
            + cd_list[: int(a * len(cd_list))]
            + hyp_list[: int(a * len(hyp_list))]
        )
        val_ids = (
            no_list[int(a * len(no_list)) : int(b * len(no_list))]
            + mi_list[int(a * len(mi_list)) : int(b * len(mi_list))]
            + sttc_list[int(a * len(sttc_list)) : int(b * len(sttc_list))]
            + cd_list[int(a * len(cd_list)) : int(b * len(cd_list))]
            + hyp_list[int(a * len(hyp_list)) : int(b * len(hyp_list))]
        )
        test_ids = (
            no_list[int(b * len(no_list)) :]
            + mi_list[int(b * len(mi_list)) :]
            + sttc_list[int(b * len(sttc_list)) :]
            + cd_list[int(b * len(cd_list)) :]
            + hyp_list[int(b * len(hyp_list)) :]
        )

        return train_ids, val_ids, test_ids

    def load_ptbxl(self, data_path, label_path, flag=None):
        """
        Loads ptb-xl data from npy files
        """
        feature_list = []
        label_list = []
        filenames = []
        subject_label = np.load(label_path)
        
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        
        if flag == "TRAIN":
            ids = self.train_ids
        elif flag == "VAL":
            ids = self.val_ids
        elif flag == "TEST":
            ids = self.test_ids
        else:
            ids = subject_label[:, 1]

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                if j + 1 in ids:
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.X[index]).float(),
            torch.tensor(self.y[index], dtype=torch.long)
        )

    def __len__(self):
        return len(self.y)


def get_ptbxl_dataloaders(root_path, batch_size=32, num_workers=4):
    """
    Create train/val/test dataloaders for PTB-XL dataset
    
    Args:
        root_path: Path to PTB-XL data directory
        batch_size: Batch size
        num_workers: Number of data loading workers
    
    Returns:
        train_loader, val_loader, test_loader, dataset_info
    """
    from torch.utils.data import DataLoader
    
    train_dataset = PTBXLLoader(root_path, flag="TRAIN")
    val_dataset = PTBXLLoader(root_path, flag="VAL")
    test_dataset = PTBXLLoader(root_path, flag="TEST")
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    dataset_info = {
        'num_channels': train_dataset.num_channels,
        'seq_len': train_dataset.max_seq_len,
        'num_classes': train_dataset.num_classes,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
    }
    
    return train_loader, val_loader, test_loader, dataset_info
