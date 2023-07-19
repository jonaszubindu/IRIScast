import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''
variational autoencoder (VAE) with a double convolution layer
Encoder -> Decoder -> Training loop
input data: spectra of shape (batch, x, lambda)
'''
# hyperparameters
latent_dim = 20 # 10 features from the feature selection 
input_dim = 480
'''
1)
'''
class Encoder(nn.Module):
    '''
    encoder learns a distribution/function Q(z|x) to better approximate the intractable marginal likelihood p(x)
    by sampling only the most relevant contributions from the latent space
    '''
    def __init__(self):
        super().__init__()
        
        # conv1(channels_in, channels_out, kernel_size=, stride=,)
        # this convolution takes in a batch of spectra, reduces each spectra to a size 240/24 (stride 12) = 20
        # and outputs 10 different vectors of length 30 based on 10 different kernel weights 
        # they are then passed through a ReLU activation function
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 40, kernel_size=6, stride=6), 
            nn.ReLU(True))
        
        # conv2 takes the 10 input channels and produces 20 output channels each of length 30/10 (stride 5) = 5
        # they are then passed through a ReLU activation function
        self.conv2 = nn.Sequential(
            nn.Conv1d(40, 20, kernel_size=4, stride=4), #kernel_size = stride = 6 for 240 lambda points
                                                        #kernel_size = stride = 12 for 480 lambda points
                                                        #changing the input dimension of the spectra lambda points changes the 
                                                        #stride and kernel size necessary to end up with the desired 
                                                        #dimensions.
            nn.ReLU(True))
        
        self.fc1 = nn.Linear(20*20, 50) # fully connected layer (Linear means wx^T + b)
        #self.fc2 = nn.Linear(50, 30)
        # reduce dimension to latent dimension and produce two outputs which will be used to approximate 
        # the posterior p(z|x) with a multivariate Gaussian N(mu,var)
        self.mu = nn.Linear(50, latent_dim)
        self.var = nn.Linear(50, latent_dim)

    def forward(self, x):
        # pass the data through the network defined above
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = F.relu( self.fc1(x) )
        #x = F.relu( self.fc2(x) )
        z_mu = self.mu(x)
        z_var = self.var(x)

        return z_mu, z_var
'''
2)
'''

#class Decoder(nn.Module):
    
#    def __init__(self):
#        super().__init__()
#        self.decoder_lin = nn.Sequential(
#            nn.Linear(latent_dim, 100),
#            nn.ReLU(True),
#            nn.Linear(100, 24 *30 ),
#            nn.ReLU(True)
#        )

#        self.unflatten = nn.Unflatten(dim=1, 
#        unflattened_size=(30, 24))

#        self.decoder_conv = nn.Sequential(
#            nn.ConvTranspose1d(30, 50, kernel_size = 5, 
#            stride=5),
#            nn.ReLU(True),
#            nn.ConvTranspose1d(50, 1, kernel_size = 2, stride=2)
#        )
        
#    def forward(self, x):
#        x = x.view(-1, latent_dim) # -1 adapts to the batch size
        #print(x.shape)
#        x = self.decoder_lin(x)
        #print(x.shape)
#        x = self.unflatten(x)
#        x = self.decoder_conv(x)
#        x = torch.sigmoid(x)
#        return x
        
class Decoder(nn.Module):
    '''
    Generates a spectra after sampleing a latent vector
    '''
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(latent_dim, 30)
        self.fc2 = nn.Linear(30, input_dim)
        #self.fc3 = nn.Linear(100, input_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        # sigmoid activation function outputs a vector whose values are between 0 and 1 (like our normalized spectra)#
        generated_spectra = torch.sigmoid(self.fc2(x))

        return generated_spectra
'''
3)
'''
class VAE(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()

        self.enc = enc
        self.dec = dec

    def forward(self, x):
        # encode
        z_mu, z_var = self.enc(x)

        # sample from the distribution having latent parameters z_mu, z_var
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        # parameterization trick for stochastic gradient descent 
        z_sample = eps.mul(std).add_(z_mu)

        # decode
        generated_spectra = self.dec(z_sample)

        return generated_spectra, z_mu, z_var
    
def add_noise(inputs, noise_factor=0.03):
    noisy = inputs+torch.randn_like(inputs) * noise_factor
    noisy = torch.clip(noisy,0.,1.)
    return noisy

