import torch
import torchvision.models as models
from torch import nn
from torch.autograd import Variable


class VariationalEncoder(nn.Module):
    def __init__(self, nc, ndf, latent_dims, device):
        super(VariationalEncoder, self).__init__()

        self.device = device

        self.nc = nc
        self.ndf = ndf
        self.latent_dims = latent_dims
        # self.conv1 = nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias=False)
        # self.batch1 = nn.BatchNorm2d(ndf)

        # self.conv2 = nn.Conv2d(ndf, ndf*2, 4, stride=2, padding=1, bias=False)
        # self.batch2 = nn.BatchNorm2d(ndf*2)

        # self.conv3 = nn.Conv2d(ndf*2, ndf*4, 4, stride=2,
        #                        padding=1, bias=False)
        # self.batch3 = nn.BatchNorm2d(ndf*4)

        # self.conv4 = nn.Conv2d(ndf*4, ndf*8, 4, stride=2,
        #                        padding=1, bias=False)
        # self.batch4 = nn.BatchNorm2d(ndf*8)

        # self.conv5 = nn.Conv2d(ndf*8, ndf*8, 4, stride=2,
        #                        padding=1, bias=False)
        # self.batch5 = nn.BatchNorm2d(ndf*8)

        self.resnet = models.resnet152(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        # modules = list(resnet.children())[:-1]      # delete the last fc layer.
        #self.resnet = nn.Sequential(*modules)

        #self.vgg.classifier[6] = nn.Linear(4096, ndf*8*4)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, ndf*8*4)
        self.linear2 = nn.Linear(ndf*8*4, latent_dims)
        self.linear3 = nn.Linear(ndf*8*4, latent_dims)

        self.N = torch.distributions.Normal(0, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    # def reparametrise(self, mu, sigma):
    #     z = mu + sigma*self.N.sample(mu.shape)
    #     return z

    def reparametrise(self, mu, logvar):
        # z = mu + sigma*self.N.sample(mu.shape)
        # return z

        std = logvar.mul(0.5).exp_()

        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        # x = x.to(device)
        # x = self.leakyrelu(self.batch1(self.conv1(x)))
        # x = self.leakyrelu(self.batch2(self.conv2(x)))
        # x = self.leakyrelu(self.batch3(self.conv3(x)))
        # x = self.leakyrelu(self.batch4(self.conv4(x)))
        # x = self.leakyrelu(self.batch5(self.conv5(x)))
        x = self.resnet(x)
        #x = torch.flatten(x, start_dim=1)
        #x = x.view(-1, self.ndf*8*4*4)
        #x = x.view(-1, 3*3*32)
        #x = F.relu(self.vgg.classifier[6](x))
        mu = self.linear2(x)
        logvar = self.linear3(x)
        z = self.reparametrise(mu, logvar)
        # # K-L divergence
        # self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, logvar


class Decoder(nn.Module):

    def __init__(self, nc, ngf, latent_dims):
        super().__init__()

        # self.decoder_lin = nn.Sequential(
        #     nn.Linear(latent_dims, 128),
        #     nn.ReLU(True),
        #     nn.Linear(128, 3 * 3 * 32),
        #     nn.ReLU(True)
        # )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(ngf*8*2, 4, 4))

        # self.decoder_conv = nn.Sequential(
        #     nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(16, 8, 3, stride=2,
        #                        padding=1, output_padding=1),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1, output_padding=1)
        # )

        self.ngf = ngf
        self.latent_dims = latent_dims
        self.leakyrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.d1 = nn.Linear(latent_dims, ngf*8*2*4*4)

        self.up1 = nn.UpsamplingBilinear2d(scale_factor=1)
        self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = nn.Conv2d(ngf*8*2, ngf*8, 3, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(ngf*8, 1.e-3)

        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.pd2 = nn.ReplicationPad2d(1)
        self.d3 = nn.Conv2d(ngf*8, ngf*4, 3, 1, bias=False)
        self.bn7 = nn.BatchNorm2d(ngf*4, 1.e-3)

        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.pd3 = nn.ReplicationPad2d(1)
        self.d4 = nn.Conv2d(ngf*4, ngf*2, 3, 1, bias=False)
        self.bn8 = nn.BatchNorm2d(ngf*2, 1.e-3)

        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.pd4 = nn.ReplicationPad2d(1)
        self.d5 = nn.Conv2d(ngf*2, ngf, 3, 1, bias=False)
        self.bn9 = nn.BatchNorm2d(ngf, 1.e-3)

        self.up5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.pd5 = nn.ReplicationPad2d(1)
        self.d6 = nn.Conv2d(ngf, nc, 3, 1)

    def forward(self, x):
        # x = self.decoder_lin(x)
        # x = self.unflatten(x)
        # x = self.decoder_conv(x)
        # x = torch.sigmoid(x)
        x = self.relu(self.d1(x))
        x = self.unflatten(x)
        x = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(x)))))
        x = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(x)))))
        x = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(x)))))
        x = self.leakyrelu(self.bn9(self.d5(self.pd4(self.up4(x)))))

        #x = x.view(-1, self.ngf*8, 4, 4)
        # x = self.unflatten(x)
        # x = F.relu(self.debatch1(self.deconv1(x)))
        # x = F.relu(self.debatch2(self.deconv2(x)))
        # x = F.relu(self.debatch3(self.deconv3(x)))
        # x = F.relu(self.debatch4(self.deconv4(x)))
        # x = self.deconv5(x)
        x = (self.d6(self.pd5(self.up5(x))))

        return x


class VariationalAutoencoder(nn.Module):
    def __init__(self, nc, ndf, ngf, latent_dims, device):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(nc, ndf, latent_dims, device)
        self.decoder = Decoder(nc, ngf, latent_dims)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        final = self.decoder(z)
        return final, mu, logvar
