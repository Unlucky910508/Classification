import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data import random_split

trainImagePath = "att_faces/Training"
testImagePath = "att_faces/Testing"

trainSet = datasets.ImageFolder(root=trainImagePath)
testSet = datasets.ImageFolder(root=testImagePath)


trainSetLen = len(trainSet)
testSetLen = len(testSet)

print("Training Set Number: ", trainSetLen)
print("Testing Set Number: ", testSetLen)


"""
for i in testSet:
    print(testSet.classes[i[1]])
    i[0].show()
"""
