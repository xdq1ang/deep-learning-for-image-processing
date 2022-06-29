import torch
import torch.nn as nn
import torch.nn.functional as F
from discriminator.utils import eightway_affinity_kld, fourway_affinity_kld
from discriminator.PPM import PPM,DensePPM


class FCDiscriminator(nn.Module):
    def __init__(self, num_classes, ndf=64):
        super(FCDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x


class EightwayASADiscriminator(nn.Module):
    def __init__(self, num_classes, ndf = 64,ppm = None):
        super(EightwayASADiscriminator, self).__init__()
        self.ppm = ppm
        self.conv1 = nn.Conv2d(8*num_classes, ndf, kernel_size=4, stride=1, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=1, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=1, padding=1)

        out = ndf*8
        self.pool_size= [2,3,4,5]
        if self.ppm == 'ppm':
            self.ppm = PPM(ndf*8,64,[1,2,3,6])
            out = ndf*8+ndf*4
        elif self.ppm == 'denseppm':
            self.ppm = DensePPM(in_channels=ndf*8,reduction_dim= 32,pool_sizes=self.pool_size)
            out = ndf*8+32*((len(self.pool_size)*len(self.pool_size)+len(self.pool_size))//2)
        
        self.classifier = nn.Conv2d(out, 1, kernel_size=4, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = eightway_affinity_kld(x)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        if self.ppm != None:
            x = self.ppm(x)
        x = self.classifier(x)
        return x

class FourwayASADiscriminator(nn.Module):
    def __init__(self, num_classes, ndf=64):
        super(FourwayASADiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(4*num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = fourway_affinity_kld(x)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    net = EightwayASADiscriminator(num_classes=19)
    input = torch.randn(1, 19, 512, 1024)
    output = net(input)
