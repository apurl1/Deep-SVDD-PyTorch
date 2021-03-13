import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class Wheel_LeNet(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 32
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 16, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(16, 32, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv4 = nn.Conv2d(64, 64, 5, bias=False, padding=2)
        self.bn4 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv5 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn5 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.conv6 = nn.Conv2d(128, 256, 5, bias=False, padding=2)
        self.bn6 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(256 * 4 * 4, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn3(x)))
        x = self.conv4(x)
        x = self.pool(F.leaky_relu(self.bn4(x)))
        x = self.conv5(x)
        x = self.pool(F.leaky_relu(self.bn5(x)))
        x = self.conv6(x)
        x = self.pool(F.leaky_relu(self.bn6(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class Wheel_LeNet_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 32
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv2d(1, 16, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(16, 32, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv4 = nn.Conv2d(64, 64, 5, bias=False, padding=2)
        self.bn4 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv5 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn5 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.conv6 = nn.Conv2d(128, 256, 5, bias=False, padding=2)
        self.bn6 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(256 * 4 * 4, self.rep_dim, bias=False)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(2, 128, 5, bias=False, padding=2)
        self.dbn1 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, bias=False, padding=2)
        self.dbn2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(64, 64, 5, bias=False, padding=2)
        self.dbn3 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
        self.dbn4 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv5 = nn.ConvTranspose2d(32, 16, 5, bias=False, padding=2)
        self.dbn5 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.deconv6 = nn.ConvTranspose2d(16, 1, 5, bias=False, padding=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn3(x)))
        x = self.conv4(x)
        x = self.pool(F.leaky_relu(self.bn4(x)))
        x = self.conv5(x)
        x = self.pool(F.leaky_relu(self.bn5(x)))
        x = self.conv6(x)
        x = self.pool(F.leaky_relu(self.bn6(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = x.view(x.size(0), int(self.rep_dim / 16), 4, 4)
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.dbn1(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.dbn2(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.dbn3(x)), scale_factor=2)
        x = self.deconv4(x)
        x = F.interpolate(F.leaky_relu(self.dbn4(x)), scale_factor=2)
        x = self.deconv5(x)
        x = F.interpolate(F.leaky_relu(self.dbn5(x)), scale_factor=2)
        x = self.deconv6(x)
        x = torch.sigmoid(x)

        return x
