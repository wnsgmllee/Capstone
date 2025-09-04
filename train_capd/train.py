import numpy as np
import os 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from FeatureDataset import FeatureDataset
from model import TransformerModel
from utils import custom_collate_fn

# 🏋️♂️ 학습 함수
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, targets, _ in dataloader:
        inputs, targets = inputs.to(device), targets.to(device).float()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Train Loss: {avg_loss:.4f}")


# 📊 테스트 함수 
def test(model, dataloader, device):
    model.eval()
    preds, gts = [], []
    total_loss = 0
    mae_loss_fn = nn.L1Loss()
    mse_loss_fn = nn.MSELoss()

    save_dir = '/data/coqls1229/repos/capd/pred_features'
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for inputs, targets, fnames in dataloader:
            inputs, targets = inputs.to(device), targets.to(device).float()
            outputs = model(inputs)

            preds.append(outputs.cpu())
            gts.append(targets.cpu())

            # 🔥 예측 결과 저장
            for output, fname in zip(outputs.cpu(), fnames):
                base_name = os.path.splitext(fname)[0]
                save_path = os.path.join(save_dir, base_name + '.npy')
                np.save(save_path, output.numpy())

            loss = mse_loss_fn(outputs, targets)
            total_loss += loss.item()

    preds = torch.cat(preds)
    gts = torch.cat(gts)

    avg_mse = total_loss / len(dataloader)
    avg_mae = mae_loss_fn(preds, gts).item()

    print(f"Test MSE Loss: {avg_mse:.4f} | MAE: {avg_mae:.4f}")
    return preds, gts

# 🚀 메인 실행 함수
def main():
    # 설정
    base_dir = '/data/jhlee39/workspace/repos/Diff-Foley/dataset/features'
    batch_size = 4
    lr = 1e-4
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 로더
    train_dataset = FeatureDataset(base_dir, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_dataset = FeatureDataset(base_dir, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

    # 모델 및 옵티마이저 세팅
    model = TransformerModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train(model, train_loader, criterion, optimizer, device)
        preds, gts = test(model, test_loader, device)

        # 추후 metric 계산 등 추가 가능

if __name__ == '__main__':
    main()
