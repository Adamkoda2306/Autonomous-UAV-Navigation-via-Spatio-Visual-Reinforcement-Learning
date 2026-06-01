import torch
import torch.nn as nn

class DoubleConv(nn.Module):

    def __init__(self,in_ch,out_ch):

        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch,out_ch,3,padding=1),
            nn.ReLU()
        )

    def forward(self,x):
        return self.net(x)


class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.d1 = DoubleConv(3,64)
        self.p1 = nn.MaxPool2d(2)

        self.d2 = DoubleConv(64,128)
        self.p2 = nn.MaxPool2d(2)

        self.d3 = DoubleConv(128,256)

        self.u1 = nn.ConvTranspose2d(256,128,2,2)
        self.c1 = DoubleConv(256,128)

        self.u2 = nn.ConvTranspose2d(128,64,2,2)
        self.c2 = DoubleConv(128,64)

        self.out = nn.Conv2d(64,1,1)

    def forward(self,x):

        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))

        d3 = self.d3(self.p2(d2))

        u1 = self.u1(d3)

        c1 = self.c1(torch.cat([u1,d2],dim=1))

        u2 = self.u2(c1)

        c2 = self.c2(torch.cat([u2,d1],dim=1))

        return torch.sigmoid(self.out(c2))