import torch
import torch.nn as nn
import torch.nn.functional as F

# Create CNN Model
class UNet_multi_task(nn.Module):
    def __init__(self):
        super(UNet_multi_task, self).__init__()

        # encoder
        self.conv1_1 = nn.Sequential(nn.Conv3d(1, 32, 3, 1, padding=1, dilation=1),
                                     nn.BatchNorm3d(32),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.3))

        self.conv1_2 = nn.Sequential(nn.Conv3d(32, 32, 3, 1, padding=1, dilation=1),
                                     nn.BatchNorm3d(32),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.3))

        self.conv2_1 = nn.Sequential(nn.Conv3d(32, 64, 3, 1, padding=1, dilation=1),
                                     nn.BatchNorm3d(64),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.3))

        self.conv2_2 = nn.Sequential(nn.Conv3d(64, 64, 3, 1, padding=1, dilation=1),
                                     nn.BatchNorm3d(64),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.3))

        self.conv3_1 = nn.Sequential(nn.Conv3d(64, 128, 3, 1, padding=1, dilation=1),
                                     nn.BatchNorm3d(128),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.3))

        self.conv3_2 = nn.Sequential(nn.Conv3d(128, 128, 3, 1, padding=1, dilation=1),
                                     nn.BatchNorm3d(128),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.3))

        self.conv4_1 = nn.Sequential(nn.Conv3d(128, 256, 3, 1, padding=1, dilation=1),
                                     nn.BatchNorm3d(256),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.3))

        self.conv4_2 = nn.Sequential(nn.Conv3d(256, 128, 3, 1, padding=1, dilation=1),
                                     nn.BatchNorm3d(128),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.3))

        # decoder
        self.upconv_5 = nn.ConvTranspose3d(128, 128, 3, 2, padding=1, dilation=1, output_padding=1)

        self.conv5_1 = nn.Sequential(nn.Conv3d(256, 128, 3, 1, padding=1, dilation=1),
                                     nn.BatchNorm3d(128),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.3))

        self.conv5_2 = nn.Sequential(nn.Conv3d(128, 64, 3, 1, padding=1, dilation=1),
                                     nn.BatchNorm3d(64),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.3))

        self.upconv_6 = nn.ConvTranspose3d(64, 64, 3, 2, padding=1, dilation=1, output_padding=1)

        self.conv6_1 = nn.Sequential(nn.Conv3d(128, 64, 3, 1, padding=1, dilation=1),
                                     nn.BatchNorm3d(64),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.3))

        self.conv6_2 = nn.Sequential(nn.Conv3d(64, 32, 3, 1, padding=1, dilation=1),
                                     nn.BatchNorm3d(32),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.3))

        self.upconv_7 = nn.ConvTranspose3d(32, 32, 3, 2, padding=1, dilation=1, output_padding=1)

        self.conv7_1 = nn.Sequential(nn.Conv3d(64, 32, 3, 1, padding=1, dilation=1),
                                     nn.BatchNorm3d(32),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.3))

        self.conv7_2 = nn.Sequential(nn.Conv3d(32, 1, 3, 1, padding=1, dilation=1),
                                     nn.BatchNorm3d(1),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.3))

        self.pool = nn.MaxPool3d(2)

        self.disps = torch.arange(0, 128, requires_grad=False).view((1, -1, 1, 1)).float().cuda()

    def forward(self, x):
        c1_1 = self.conv1_1(x)
        c1_2 = self.conv1_2(c1_1)
        p1 = self.pool(c1_2)

        c2_1 = self.conv2_1(p1)
        c2_2 = self.conv2_2(c2_1)
        p2 = self.pool(c2_2)

        c3_1 = self.conv3_1(p2)
        c3_2 = self.conv3_2(c3_1)
        p3 = self.pool(c3_2)

        c4_1 = self.conv4_1(p3)
        c4_2 = self.conv4_2(c4_1)

# upsampling start
# first branch

        b5 = self.upconv_5(c4_2)
        b5_concat = torch.cat((b5, c3_2), 1)

        b5_1 = self.conv5_1(b5_concat)
        b5_2 = self.conv5_2(b5_1)

        b6 = self.upconv_6(b5_2)
        b6_concat = torch.cat((b6, c2_2), 1)

        b6_1 = self.conv6_1(b6_concat)
        b6_2 = self.conv6_2(b6_1)

        b7 = self.upconv_7(b6_2)
        b7_concat = torch.cat((b7, c1_2), 1)

        b7_1 = self.conv7_1(b7_concat)
        b7_2 = self.conv7_2(b7_1)

# second branch

        c5 = self.upconv_5(c4_2)
        c5_concat = torch.cat((c5, c3_2), 1)

        c5_1 = self.conv5_1(c5_concat)
        c5_2 = self.conv5_2(c5_1)

        c6 = self.upconv_6(c5_2)
        c6_concat = torch.cat((c6, c2_2), 1)

        c6_1 = self.conv6_1(c6_concat)
        c6_2 = self.conv6_2(c6_1)

        c7 = self.upconv_7(c6_2)
        c7_concat = torch.cat((c7, c1_2), 1)

        c7_1 = self.conv7_1(c7_concat)
        c7_2 = self.conv7_2(c7_1)

        # if infer:
        #     last = b7_2 + c7_2


        return b7_2, c7_2 # ,last




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet_multi_task().to(device)
# print(net)
# torch.cuda.empty_cache()

from torchsummary import summary
summary(net, (1,128,128,128))