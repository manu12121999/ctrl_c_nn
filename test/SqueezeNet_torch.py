import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import urllib.request
from PIL import Image
from torchvision import transforms


class Fire(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        )


class SqueezeNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(512, self.num_classes, kernel_size=1, groups=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print("CONV 1", self.features[0](x))
        x = self.features(x)
        print("features", x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


if __name__ == "__main__":

    # get sample image
    # urllib.request.urlretrieve("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")

    # get weights:
    urllib.request.urlretrieve("https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth", "archive.pth")

    # get class names:
    urllib.request.urlretrieve("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")

    # # Loading model weights
    model = SqueezeNet()
    model.load_state_dict(torch.load("archive.pth"))
    model.eval()

    input_image = Image.open("dog.png").convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image).unsqueeze(0)

    #if torch.cuda.is_available():
    #    input_tensor = input_tensor.to('cuda')
    #    model.to('cuda')

    with torch.no_grad():
        start = time.time()
        output = model(input_tensor)
        print("INFERENCE", time.time() - start)

    probabilities = F.softmax(output[0], dim=0)


    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    top5_prob, top5_catid = torch.topk(probabilities, 5)

    print("RESULTS\n")
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())