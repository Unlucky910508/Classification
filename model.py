from torch import nn
import torchvision


class ModelA(nn.Module):
    def __init__(self) -> None:
        super(ModelA, self).__init__()
        self.model = torchvision.models.resnet18(
            torchvision.models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(512, 530)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = ModelA()
