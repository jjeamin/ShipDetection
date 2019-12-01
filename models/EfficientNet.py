import torch.nn as nn
from efficientnet_pytorch import EfficientNet

#class BiFPN(nn.Module):
#    def __init__(self):



class EfficientDetv1(nn.Module):
    def __init__(self, name):
        super(EfficientDetv1, self).__init__()

        self.name = name
        self.feature = EfficientNet.from_pretrained('efficientnet-b1')

    def forward(self, x):
        x, c = self.feature.extract_features(x)

        return x