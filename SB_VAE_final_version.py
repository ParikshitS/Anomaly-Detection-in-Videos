import torch
import torch.utils.data
from torch import nn#, optim
from torch.nn import functional as F
import sys

import math
from tqdm import tqdm
import numpy as np
import pickle
import random
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
# from   torch.autograd import Variable



torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)



train_images = np.load('shanghaiTrainImages.npy')
num_epochs = int(sys.argv[1])

print('Length of training images is {}'.format(len(train_images)))

def make_cuboids(frames, temporal_size, spatial_size):
  """Create cuboids of video samples.
  Currenlty non-overlapping, non-skipping cuboids are created

  Parameters
  ----------
  frames : list
      each list element has individual frame of a video
  temporal_size : int
      indicates how many contiguous frames to concatinate to construct sample
  spatial_size : int
      spatial dimension to break the frames into (same size used for height and width)

  Returns
  -------
  numpy array
      Np array of cuboids of shape (_, spatial_size, spatial_size, temporal_size, 3) 
      currently shape: (_, 5, 32, 32, 3)
  """
  #First create spatial cuts of 32x32
  frames_spatial = [0] * len(frames)

  for f, frame in enumerate(frames):
    frames_spatial[f] = []
    for h in range(0, frame.shape[0]-spatial_size+1, spatial_size):
      for w in range(0, frame.shape[1]-spatial_size+1, spatial_size):
        frames_spatial[f].append(frame[h:h+spatial_size, w:w+spatial_size])
  #Now join these 32x32 frames temporaly
  spatio_temporal_block = []
  for i in range(0, len(frames_spatial)-temporal_size+1, temporal_size):
    a = [frames_spatial[i+t] for t in range(temporal_size)]
    for z in range(16):  #Why 16? --> 128x128 when divided into 32x32, gives 16 such. So hardcoded to work only on 32x32.
      spatio_temporal_block.append([b[z] for b in a])
  return np.array(spatio_temporal_block)

x = make_cuboids(train_images, 20, 32)
print(x.shape)
x = x.astype('float64')
x *= 255.0/x.max()
train_cuboids = torch.from_numpy(x)
train_cuboids = train_cuboids.permute(0, 4, 1, 2, 3)
batch_size = 1200
train_loader = torch.utils.data.DataLoader(train_cuboids, shuffle=True, batch_size=batch_size, num_workers=4, drop_last=True)

use_cuda = torch.cuda.is_available()
#device = torch.device("cuda:0" if use_cuda else "cpu")
device = torch.device("cuda" if use_cuda else "cpu")
cudnn.benchmark = True


class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).to(device)
        self.noise = self.noise.type(torch.cuda.FloatTensor)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x
    
def Beta_fn(a, b):
    return torch.exp(a.lgamma() + b.lgamma() - (a+b).lgamma())


def calc_kl_divergence(a, b, prior_alpha, prior_beta):
    kl = 0
    for i in range(1,11,1):
        kl += 1./(i+a*b) * Beta_fn(i/a, b)
    kl *= (prior_beta-1)*b
    psi_b_taylor_approx = torch.log(b) - 1./(2 * b) - 1./(12 * b**2)
    kl += (a-prior_alpha)/a * (-0.57721 - psi_b_taylor_approx - 1/b)
    kl += torch.log(a*b) + torch.log(Beta_fn(prior_alpha, prior_beta))
    kl += -(b-1)/b

    return torch.sum(kl)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class UnFlatten(nn.Module):
    def forward(self, input, size=8):
        return input.view(input.size(0), 512, 1, 2, 2)
    
class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale = scale
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        return x
    
class SB_VAE(nn.Module):
    def __init__(self, lr = 0.002, n_gmm = 10, latent_dim=2048):
        super(SB_VAE, self).__init__()
        
        
        
        self.mean = nn.Linear(2048, latent_dim)
        self.logvar = nn.Linear(2048, latent_dim)

        layers = []
        layers += [nn.Linear(2048, 200)]
        layers += [nn.Softplus()]
        layers += [nn.Linear(200, n_gmm)]
        layers += [nn.Softplus()]
        self.b = nn.Sequential(*layers)
        self.unflatten = nn.Linear(latent_dim, 2048)
        
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool3d(2, stride=2),
            nn.Conv3d(64, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool3d(2, stride=2),
            nn.Conv3d(128, 256, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool3d(2, stride=2),
            nn.Conv3d(256, 512, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool3d(2, stride=2),
            Flatten()
            )
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            Interpolate(scale=2, mode='trilinear'),
            nn.Conv3d(512, 256, 3, stride=1, padding=(2, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            Interpolate(scale=2, mode='trilinear'),
            nn.Conv3d(256, 128, 3, stride=1, padding=(2, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            Interpolate(scale=2, mode='trilinear'),
            nn.Conv3d(128, 64, 3, stride=1, padding=(1, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            Interpolate(scale=2, mode='trilinear'),
            nn.Conv3d(64, 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(3, 3, 3, stride=(2, 1, 1), padding= 1),
            nn.LeakyReLU(0.2, inplace=True)#,
            # nn.Sigmoid()
            )
        
        self.pi = torch.nn.Parameter(data=torch.ones(n_gmm)/n_gmm, requires_grad=False)
        self.mu = torch.nn.Parameter(data=torch.randn(n_gmm, latent_dim), requires_grad=False)
        self.var = torch.nn.Parameter(data=torch.ones(n_gmm, latent_dim), requires_grad=False)

        self.lr = lr
    def encode(self, x):
        x1 = self.encoder(x)
        mu = self.mean(x1)
        logvar = self.logvar(x1)
        beta_b = self.b(x1)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        z = eps * std + mu
        return z, mu, logvar, beta_b
    
    def decode(self, z):
        z = self.unflatten(z)
        xx = self.decoder(z)
        return xx.view(-1, 3, 20, 32, 32)
    
    def gmm(self, z):
        temp_p_c_z = torch.exp(torch.sum( torch.log(self.pi.unsqueeze(0).unsqueeze(-1)+1e-10) - 0.5* torch.log(2*math.pi*self.var.unsqueeze(0)+1e-10) - (z.unsqueeze(-2)-self.mu.unsqueeze(0)).pow(2)/(2*self.var.unsqueeze(0)+ 1e-10), -1)) + 1e-10
        return temp_p_c_z / torch.sum(temp_p_c_z, -1).unsqueeze(-1)

    def forward(self, x, lr):
        self.lr = lr
        z, mu, logvar, beta_b = self.encode(x)
        beta_a = torch.Tensor(beta_b.shape).zero_().to(device)+1
        gamma, remaining_gamma = self.reparameterizebeta(beta_a, beta_b)
        self.compute_gmm_params(z, gamma)
        gamma_l = self.gmm(z)
        xx = self.decode(z)
        return xx, z, gamma, mu, logvar, gamma_l, beta_a, beta_b
    
    def compute_gmm_params(self, z, gamma):
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(-2), 0)/(torch.sum(gamma, 0).unsqueeze(-1)+1e-10)
        var = torch.sum(gamma.unsqueeze(-1) * (z.unsqueeze(-2) - mu.unsqueeze(0)).pow(2), 0)/(torch.sum(gamma, 0).unsqueeze(-1)+1e-10)

        lr = self.lr * 1.05
        self.var.data = (1-lr) * self.var.data + lr * var.clone().data
        self.mu.data = (1-lr) * self.mu.data + lr * mu.clone().data
        
    def reparameterizebeta(self, a, b):
        uniform_samples = torch.Tensor(b.shape).uniform_(0.01, 0.99)
        v_samples = (1 - (uniform_samples.to(device) ** (1 / b))) ** (1 / a)
        remaining_stick = torch.Tensor(a.shape[0], 1).zero_().to(device)+ 1
        stick_segment = torch.Tensor(a.shape).zero_().to(device)
        for i in range(a.shape[1]):
            stick_segment[:, i] = (v_samples[:, i].unsqueeze(-1) * remaining_stick).squeeze(-1)
            remaining_stick = remaining_stick.clone() * (1 - v_samples[:, i]).unsqueeze(-1)
        stick_segment = stick_segment/torch.sum(stick_segment, 1).unsqueeze(-1)
        return stick_segment, remaining_stick
    
# model = SB_VAE(lr=0.002).to(device)
model = SB_VAE(lr=0.002)
if torch.cuda.device_count() > 1:
  print("Using", torch.cuda.device_count(), "GPUs")
  model = nn.DataParallel(model)
model.to(device)
optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=0.002)

train_loss = []
def train(epoch):
    stored_mu = []
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        lr = 0.002
        if epoch>=0:
            lr = max(0.002 * 0.9 ** (np.floor(epoch) / 40.0), 0.000001)
            lr = lr * ((np.cos((epoch / (math.pi * 2))) + 1) / 2)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        data = data.float().to(device)

        optimizer.zero_grad()
        recon_batch, z, gamma, mu, logvar, gamma_l, a, b = model(data, lr=lr)
        # print(recon_batch.shape, data.shape)

        BCE = F.mse_loss(recon_batch, data, reduction='mean') * 1
        for batchs in mu:
          stored_mu.append(batchs)

        KLD = torch.sum(0.5 * gamma.unsqueeze(-1) * (
                        torch.log((torch.zeros_like(gamma.unsqueeze(-1)) + 2) * math.pi+1e-10) + torch.log(
                    model.module.var.unsqueeze(0)+1e-10) + torch.exp(logvar.unsqueeze(-2)) / (model.module.var.unsqueeze(0)+1e-10) + (
                                    mu.unsqueeze(-2) - model.module.mu.unsqueeze(0)).pow(2) / (model.module.var.unsqueeze(0)+1e-10)),
                            [-1, -2])
        KLD -= 0.5 * torch.sum(logvar + 1, -1)
        KLD += torch.sum(torch.log(gamma+1e-10) * gamma, -1)
        KLD -= torch.sum(torch.log(gamma_l+1e-10) * gamma, -1)
        KLD = torch.mean(KLD)
        prior_alpha = torch.Tensor(1).zero_().to(device) + 1
        prior_beta = torch.Tensor(1).zero_().to(device) + 3
        SBKL = calc_kl_divergence(a, b, prior_alpha, prior_beta)/ data.shape[0]
        loss = BCE + KLD + SBKL*0.005

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / (len(train_loader.dataset))))
    return stored_mu

for i in tqdm(range(num_epochs)):
    stored_mu = train(i)
    
num_epochs = [i for i in range(len(train_loss))]

plt.plot(num_epochs, train_loss)
plt.title('Train Loss Plot')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss Value')
plt.savefig('Train Loss Plot')

avg_mu = sum(stored_mu) / len(stored_mu)

print('Training Completed')

def load_test_data(PIK):
    '''
    for ped1 its both train and test togrthrt
    '''
    with open(PIK, "rb") as f:
        test_images_loaded = pickle.load(f)
        test_gt_images_loaded = pickle.load(f)
    # return test_images_loaded
    return test_images_loaded, test_gt_images_loaded


test_images = np.load('shanghaiTestImages.npy')
test_images_gt = np.load('shanghaiLabels.npy')


test_images = make_cuboids(test_images, 20, 32)
label_test_images_np = make_cuboids(test_images_gt, 20, 32)


test_cuboids = []
anomaly_tracker = [] #1 if anomaly is present; 0 if sample is normal
for i in range(len(label_test_images_np)):
  if np.count_nonzero(label_test_images_np[i]) > 8192:
    anomaly_tracker.append(1)
    test_cuboids.append(test_images[i])
  else:
      anomaly_tracker.append(0)
      test_cuboids.append(test_images[i])

     

test_cuboids_np = np.array(test_cuboids)
test_cuboids_np = test_cuboids_np.astype('float64')
test_cuboids_np *= 255.0/test_cuboids_np.max()
test_cuboids_t = torch.from_numpy(test_cuboids_np)
test_cuboids_t = test_cuboids_t.permute(0, 4, 1, 2, 3)

batch_size = 1
test_dataloader = torch.utils.data.DataLoader(test_cuboids_t, shuffle=False, batch_size=batch_size, num_workers=4, drop_last=True)

print('Test dataloader loaded')

model.eval()

def test():
    distances = []
    with torch.no_grad():
        for i, X in enumerate(test_dataloader):
            lr = 0.002
            recon_batch, z, gamma, mu, logvar, gamma_l, a, b = model(X.float().to(device), lr)
            for batchs in mu:
                distances.append(torch.dist(batchs, avg_mu, 2))
    return distances

test_distances = test()
test_distances = [ele.tolist() for ele in test_distances]
print('Done with testing')

with open('anomalies_tracker_PED1.pickle', 'wb') as f:
  pickle.dump(anomaly_tracker, f)
  

np.save('SB_VAE_PED1_ ' +str(num_epochs)+ '.npy', np.array(test_distances))
print('SB results saved')