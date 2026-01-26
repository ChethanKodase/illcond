import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from activations import Sin



#device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)

class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=1024, z_dim=256):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32*2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32*2, 64*2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64*2, 128*2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128*2, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128*2, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128*2, 64*2, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64*2, 32*2, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32*2, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z.to(device)
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu.to(device), logvar.to(device))
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h.to(device))
        z = self.fc3(z)
        #print('z.shape', z.shape)
        return self.decoder(z), mu, logvar




class VAE_big(nn.Module):
    def __init__(self, device, image_channels=3, h_dim=1024, z_dim=256):
        super(VAE_big, self).__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32*2, kernel_size=4, stride=2),
            nn.ReLU(),
            # Additional convolutional layers after the first layer
            nn.Conv2d(32*2, 32*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32*2, 32*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32*2, 32*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # till here
            nn.Conv2d(32*2, 64*2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64*2, 128*2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128*2, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128*2, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128*2, 64*2, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64*2, 32*2, kernel_size=6, stride=2),
            nn.ReLU(),
            # Additional convolutional layers after the last layer
            nn.ConvTranspose2d(32*2, 32*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32*2, 32*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32*2, 32*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # till here
            nn.ConvTranspose2d(32*2, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z.to(self.device)
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu.to(self.device), logvar.to(self.device))
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        lz, mu, logvar = self.bottleneck(h.to(self.device))
        z = self.fc3(lz)
        #print('z.shape', z.shape)
        return self.decoder(z), mu, logvar, lz
    

class VAE_big_b(nn.Module):
    def __init__(self, device, image_channels=3, h_dim=1024, z_dim=256):
        super(VAE_big_b, self).__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32*2, kernel_size=4, stride=2),
            nn.BatchNorm2d(32*2),
            nn.ReLU(),
            # Additional convolutional layers after the first layer
            nn.Conv2d(32*2, 32*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32*2),
            nn.ReLU(),
            nn.Conv2d(32*2, 32*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32*2),
            nn.ReLU(),
            nn.Conv2d(32*2, 32*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32*2),
            nn.ReLU(),
            # till here
            nn.Conv2d(32*2, 64*2, kernel_size=4, stride=2),
            nn.BatchNorm2d(64*2),
            nn.ReLU(),
            nn.Conv2d(64*2, 128*2, kernel_size=4, stride=2),
            nn.BatchNorm2d(128*2),
            nn.ReLU(),
            nn.Conv2d(128*2, 256, kernel_size=4, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128*2, kernel_size=5, stride=2),
            nn.BatchNorm2d(128*2),
            nn.ReLU(),
            nn.ConvTranspose2d(128*2, 64*2, kernel_size=5, stride=2),
            nn.BatchNorm2d(64*2),
            nn.ReLU(),
            nn.ConvTranspose2d(64*2, 32*2, kernel_size=6, stride=2),
            nn.BatchNorm2d(32*2),
            nn.ReLU(),
            # Additional convolutional layers after the last layer
            nn.ConvTranspose2d(32*2, 32*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32*2),
            nn.ReLU(),
            nn.ConvTranspose2d(32*2, 32*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32*2),
            nn.ReLU(),
            nn.ConvTranspose2d(32*2, 32*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32*2),
            nn.ReLU(),
            # till here
            nn.ConvTranspose2d(32*2, image_channels, kernel_size=6, stride=2),
            nn.BatchNorm2d(image_channels),
            nn.Sigmoid(),
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z.to(self.device)
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu.to(self.device), logvar.to(self.device))
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h.to(self.device))
        z = self.fc3(z)
        #print('z.shape', z.shape)
        return self.decoder(z), mu, logvar, z





class VAE_big_nobn(nn.Module):
    def __init__(self, device, image_channels=3, h_dim=1024, z_dim=256):
        super(VAE_big_nobn, self).__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32*2, kernel_size=4, stride=2),
            nn.BatchNorm2d(32*2),
            nn.ReLU(),
            # Additional convolutional layers after the first layer
            nn.Conv2d(32*2, 32*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32*2, 32*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32*2, 32*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # till here
            nn.Conv2d(32*2, 64*2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64*2, 128*2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128*2, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128*2, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128*2, 64*2, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64*2, 32*2, kernel_size=6, stride=2),
            nn.BatchNorm2d(32*2),
            nn.ReLU(),
            # Additional convolutional layers after the last layer
            nn.ConvTranspose2d(32*2, 32*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32*2, 32*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32*2, 32*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # till here
            nn.ConvTranspose2d(32*2, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z.to(self.device)
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu.to(self.device), logvar.to(self.device))
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h.to(self.device))
        z = self.fc3(z)
        #print('z.shape', z.shape)
        return self.decoder(z), mu, logvar, z




class VAE_small(nn.Module):
    def __init__(self, device, image_channels=3, h_dim=1024, z_dim=256):
        super(VAE_small, self).__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32*2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32*2, 64*2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64*2, 128*2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128*2, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128*2, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128*2, 64*2, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64*2, 32*2, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32*2, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z.to(self.device)
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu.to(self.device), logvar.to(self.device))
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h.to(self.device))
        z = self.fc3(z)
        #print('z.shape', z.shape)
        return self.decoder(z), mu, logvar




class VAE_big_layered(nn.Module):
    def __init__(self, image_channels=3, h_dim=1024, z_dim=256):
        super(VAE_big_layered, self).__init__()
        self.encoder_6 = nn.Sequential(
            nn.Conv2d(image_channels, 32*2, kernel_size=4, stride=2),
            nn.ReLU()
        )
        
        self.encoder_5 = nn.Sequential(
            nn.Conv2d(32*2, 32*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.encoder_4 = nn.Sequential(
            nn.Conv2d(32*2, 32*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.encoder_3 = nn.Sequential(
            nn.Conv2d(32*2, 32*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.encoder_2 = nn.Sequential(
            nn.Conv2d(32*2, 64*2, kernel_size=4, stride=2),
            nn.ReLU()
        )

        self.encoder_1 = nn.Sequential(
            nn.Conv2d(64*2, 128*2, kernel_size=4, stride=2),
            nn.ReLU()
        )


        self.encoder_0 = nn.Sequential(
            nn.Conv2d(128*2, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )


        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder_0 = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128*2, kernel_size=5, stride=2),
            nn.ReLU()
        )

        self.decoder_1 = nn.Sequential(
            nn.ConvTranspose2d(128*2, 64*2, kernel_size=5, stride=2),
            nn.ReLU()
        )


        self.decoder_2 = nn.Sequential(
            nn.ConvTranspose2d(64*2, 32*2, kernel_size=6, stride=2),
            nn.ReLU()
        )

        self.decoder_3 = nn.Sequential(
            nn.ConvTranspose2d(32*2, 32*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.decoder_4 = nn.Sequential(
            nn.ConvTranspose2d(32*2, 32*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.decoder_5 = nn.Sequential(
            nn.ConvTranspose2d(32*2, 32*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.decoder_6 = nn.Sequential(
            nn.ConvTranspose2d(32*2, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z.to(device)
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu.to(device), logvar.to(device))
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder_6(x)
        h = self.encoder_5(h)
        h = self.encoder_4(h)
        h = self.encoder_3(h)
        h = self.encoder_2(h)
        h = self.encoder_1(h)
        h = self.encoder_0(h)
        z, mu, logvar = self.bottleneck(h.to(device))
        z = self.fc3(z)
        z = self.decoder_0(z)
        z = self.decoder_1(z)
        z = self.decoder_2(z)
        z = self.decoder_3(z)
        z = self.decoder_4(z)
        z = self.decoder_5(z)
        #z = self.decoder_6(z)

        #print('z.shape', z.shape)
        return self.decoder_6(z), mu, logvar



class VAE_big_consti(nn.Module):
    def __init__(self, image_channels=3, h_dim=1024, z_dim=256):
        super(VAE_big_consti, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32*2, kernel_size=4, stride=2),
            nn.ReLU(),
            # Additional convolutional layers after the first layer
            nn.Conv2d(32*2, 32*2, kernel_size=5, stride=1 , padding=1),
            nn.ReLU(),
            nn.Conv2d(32*2, 32*2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32*2, 32*2, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            # till here
            nn.Conv2d(32*2, 64*2, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv2d(64*2, 128*2, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(128*2, 256, kernel_size=4, stride=1),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128*2, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128*2, 64*2, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64*2, 32*2, kernel_size=5, stride=1),
            nn.ReLU(),
            # Additional convolutional layers after the last layer
            nn.ConvTranspose2d(32*2, 32*2, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32*2, 32*2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32*2, 32*2, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            # till here
            nn.ConvTranspose2d(32*2, image_channels, kernel_size=4, stride=2),
            nn.Sigmoid()
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z.to(device)
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu.to(device), logvar.to(device))
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h.to(device))

        z = self.fc3(z)
        return self.decoder(z), mu, logvar