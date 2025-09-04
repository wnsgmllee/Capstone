import os
import torch
from torch.utils.data import Dataset

class FeatureDataset(Dataset):
    def __init__(self, base_dir, mode='train'):
        assert mode in ['train', 'test'], "mode must be 'train' or 'test'"
        self.mode = mode
        self.input_dir = os.path.join(base_dir, f"frame_last_mask_{mode}set")
        self.label_dir = os.path.join(base_dir, f"{mode}set")

        self.filenames = sorted([f for f in os.listdir(self.input_dir) if f.endswith('.pt')])

        assert len(self.filenames) > 0, f"No .pt files found in {self.input_dir}"

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        input_path = os.path.join(self.input_dir, fname)
        label_path = os.path.join(self.label_dir, fname)

        input_data = torch.load(input_path)
        label_data = torch.load(label_path)

        input_feature = input_data['frame_feature'].squeeze(0)  # [1, 32, 512] → [32, 512]
        label_feature = label_data['frame_feature'].squeeze(0)

        return input_feature, label_feature, fname


'''

train.py 사용 예시

from dataloader import FeatureDataset
from torch.utils.data import DataLoader

base_dir = '/data/jhlee39/workspace/repos/Diff-Foley/dataset/features'
train_dataset = FeatureDataset(base_dir, mode='train')
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

test_dataset = FeatureDataset(base_dir, mode='test')
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

'''
