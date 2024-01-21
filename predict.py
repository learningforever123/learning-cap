import torch
import cv2
from model.dunet import PAttUNet
from torchvision import transforms
device = "cuda" if torch.cuda.is_available() else "cpu"
from torchsummary import summary
#
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net =PAttUNet(3,2)
net.load_state_dict(torch.load('./checkpoints/best_miou1.pt', map_location=device))
net.to(device)

# 测试模式
net.eval()
with torch.no_grad():
    img = cv2.imread('./predict/000.png')
    # img = Image.open('./predict/000.png')  # 读取预测的图片
    # img = img.convert("L")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(img)  # 预处理
    img = torch.unsqueeze(img, dim=0)  # 增加batch维度
    print(img.shape)
    pred = net(img.to(device))  # 网络预测
    print(pred.shape)

    out = torch.argmax(pred,dim=1)  # (1, 256, 256)
    out = torch.squeeze(out, dim=0)  # (256, 256)
    out = out.unsqueeze(dim=0)  # (1, 256, 256)
    print(set((out).reshape(-1).tolist()))
    out = (out).permute((1, 2, 0)).cpu().detach().numpy()*255  # (256, 256, 1)
    print(out.shape)
    # pred = np.uint8(pred)  # 转为图片的形式
    cv2.imwrite('./predict/re000472.png', out)  # 保存图片

