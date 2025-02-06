import torch
from torchvision import transforms
from PIL import Image
import os

# 加载模型
class EyeDiseaseModel(nn.Module):
    def __init__(self, num_classes=7):
        super(EyeDiseaseModel, self).__init__()
        self.left_net = models.resnet50(pretrained=False)
        self.right_net = models.resnet50(pretrained=False)
        self.left_net.fc = nn.Linear(self.left_net.fc.in_features, num_classes)
        self.right_net.fc = nn.Linear(self.right_net.fc.in_features, num_classes)

    def forward(self, left_img, right_img):
        left_output = self.left_net(left_img)
        right_output = self.right_net(right_img)
        return (left_output + right_output) / 2

model = EyeDiseaseModel()
model.load_state_dict(torch.load('eye_disease_model.pth'))
model.eval()

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 推理函数
def predict(left_img_path, right_img_path):
    left_image = Image.open(left_img_path).convert('RGB')
    right_image = Image.open(right_img_path).convert('RGB')

    left_image = transform(left_image).unsqueeze(0)
    right_image = transform(right_image).unsqueeze(0)

    with torch.no_grad():
        output = model(left_image, right_image)
        probabilities = torch.sigmoid(output)
        return probabilities

# 示例推理
left_img_path = 'test_left.jpg'
right_img_path = 'test_right.jpg'
probabilities = predict(left_img_path, right_img_path)
print(probabilities)