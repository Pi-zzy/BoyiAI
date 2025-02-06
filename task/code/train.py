import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import pandas as pd
from PIL import Image
import os

# 定义数据集类
class EyeDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_excel(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        left_img_name = os.path.join(self.img_dir, self.annotations.iloc[idx, 3])
        right_img_name = os.path.join(self.img_dir, self.annotations.iloc[idx, 4])
        left_image = Image.open(left_img_name).convert('RGB')
        right_image = Image.open(right_img_name).convert('RGB')

        labels = self.annotations.iloc[idx, 7:].values.astype('float32')

        if self.transform:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)

        return left_image, right_image, labels

# 定义模型
class EyeDiseaseModel(nn.Module):
    def __init__(self, num_classes=7):
        super(EyeDiseaseModel, self).__init__()
        self.left_net = models.resnet50(pretrained=True)
        self.right_net = models.resnet50(pretrained=True)
        self.left_net.fc = nn.Linear(self.left_net.fc.in_features, num_classes)
        self.right_net.fc = nn.Linear(self.right_net.fc.in_features, num_classes)

    def forward(self, left_img, right_img):
        left_output = self.left_net(left_img)
        right_output = self.right_net(right_img)
        return (left_output + right_output) / 2

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
dataset = EyeDataset(csv_file='Traning_Dataset.xlsx', img_dir='Training_Datasets', transform=transform)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# 初始化模型、损失函数和优化器
model = EyeDiseaseModel()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (left_images, right_images, labels) in enumerate(train_loader):
        left_images = left_images.to(device)
        right_images = right_images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(left_images, right_images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
            running_loss = 0.0

# 保存模型
torch.save(model.state_dict(), 'eye_disease_model.pth')