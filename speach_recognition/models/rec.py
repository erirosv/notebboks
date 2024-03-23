#!/usr/bin/env python3
import time
import torch
from torch import nn
import torch.nn.functional as F
#from preprocess import CHARSET
import torch.optim as optim
from torchaudio.models import Conformer

class Rec(nn.Module):
    def __init__(self, expand=1):
        super().__init__()
        CONST_64 = 64
        HCOSNT = 144 # 512, 256, 80
        self.expand = expand
        self.encoder = nn.Sequential(
            nn.Conv2d(1, CONST_64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(CONST_64, CONST_64, kernel_size=3, stride=2),
            nn.ReLU()
        )
        self.linear = nn.Sequential(
            nn.Linear(CONST_64*(((80 - 1) // 2 - 1) // 2), HCOSNT),
            nn.Dropout(0.1)
        )


def test():
    r = Rec()
    print(r)


if __name__ == "__main__":
    test()