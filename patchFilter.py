import torch
import torch.nn as nn
import numpy as np



class AutoEncoder8(nn.Module):

    def __init__(self):
        super(AutoEncoder8, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=8, stride=2, padding=1),    # batch, 8, 24, 24
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),    # batch, 16, 12, 12
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),    # batch, 32, 6, 6
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),    # batch, 16, 12, 12
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),    # batch, 16, 24, 24
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=8, stride=2, padding=1),    # batch, 3, 52, 52
            nn.Tanh(),
        )

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x


class AutoEncoder16(nn.Module):

    def __init__(self):
        super(AutoEncoder16, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=2, stride=2, padding=1),    # batch, 8, 14, 14
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=2, stride=2, padding=1),    # batch, 16, 8, 8
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=2, padding=1),    # batch, 32, 5, 5
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=1),    # batch, 16, 8, 8
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2, padding=1),    # batch, 8, 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=2, stride=2, padding=1),    # batch, 3, 26, 26
            nn.Tanh(),
        )

    def forward(self, x):

        x = self.encoder(x)
        #print(x.shape)
        x = self.decoder(x)
        #print(x.shape)

        return x


class AutoEncoder32(nn.Module):

    def __init__(self):
        super(AutoEncoder32, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=2, stride=1, padding=1),    # batch, 8, 14, 14
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=2, stride=2, padding=1),    # batch, 16, 8, 8
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=2, padding=1),    # batch, 32, 5, 5
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=1),    # batch, 16, 8, 8
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2, padding=1),    # batch, 8, 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=2, stride=1, padding=1),    # batch, 3, 13, 13
            nn.Tanh(),
        )

    def forward(self, x):

        x = self.encoder(x)
        #print(x.shape)
        x = self.decoder(x)
        #print(x.shape)

        return x



class NutNet(object):
    def __init__(self, inp_dim=416, device='cuda', box_num=8, thresh1=0.125, thresh2=0.2):
        self.thresh1=thresh1
        self.thresh2=thresh2
        self.inp_dim = inp_dim
        self.device = device
        self.box_num = box_num
        self.box_length = inp_dim//self.box_num
        if box_num == 8:
            self.ae = AutoEncoder8().to(device)
            self.ae.load_state_dict(torch.load("./ae_weights/n_52.pth"))
        elif box_num == 16:
            self.ae = AutoEncoder16().to(device)
            self.ae.load_state_dict(torch.load("./ae_weights/n_26.pth"))
        elif box_num == 32:
            self.ae = AutoEncoder32().to(device)
            self.ae.load_state_dict(torch.load("./ae_weights/n_13.pth"))
        self.loss = nn.MSELoss(reduction='none')

    def __call__(self, img0):
        #print(img0.shape)
        #print([self.inp_dim, self.box_num, self.box_length])
        img = img0*2-1

        img = img.unfold(2, self.box_length, self.box_length).unfold(3, self.box_length, self.box_length)
        img = img.permute(0, 2, 3, 4, 5, 1).contiguous().view(-1, self.box_length, self.box_length, 3)
        img = img.permute(0, 3, 1, 2)
        #print(img.shape)
        output = self.ae(img)
        #input(output.shape)

        loss = torch.mean(self.loss(output, img), dim=(1,2,3))
        mask1 = ((loss>self.thresh1).float().unsqueeze(1).unsqueeze(2).unsqueeze(3).expand((self.box_num*self.box_num, 3, self.box_length, self.box_length)))
        delta = torch.abs(output-img)
        mask2 = (delta.sum(dim=[1])>self.thresh2).float().unsqueeze(1).expand(-1, 3, -1, -1)
        mask = mask1*mask2
        img2 = torch.where(mask==0, img, 0)

        img2 = img2.unsqueeze(0).view(1, self.box_num, self.box_num, 3, self.box_length, self.box_length)#
        img2 = img2.permute(0, 3, 1, 4, 2, 5).contiguous().view(1, 3, self.inp_dim, self.inp_dim)
        img2 = torch.clamp(0.5*(img2+1), 0, 1)
        return img2

def nutnet_getter(inp_dim=416, device='cuda', box_num=8, thresh1=0.125, thresh2=0.2):
    return NutNet(inp_dim=inp_dim, device=device, box_num=box_num, thresh1=thresh1, thresh2=thresh2)
