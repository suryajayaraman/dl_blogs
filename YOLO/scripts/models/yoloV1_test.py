import torch
from yoloV1 import yoloV1


if __name__ == "__main__":
    model = yoloV1(splitSize=7, numBoxes=2, numClasses=20)
    testInput = torch.rand(2,3,448,448)
    output = model(testInput)
    print(output.shape)