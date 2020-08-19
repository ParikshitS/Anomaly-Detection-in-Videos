import os
import cv2
import time
import copy
import pickle
import warnings
from tqdm import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from   torch.autograd import Variable
import torch.backends.cudnn as cudnn
import sys

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

PIK = "training_lists_UCSD1.dat"
with open(PIK, "rb") as f:
    opticalflow_images = pickle.load(f)
    train_images = pickle.load(f)

    
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
train_cuboids = torch.from_numpy(x)
train_cuboids = train_cuboids.permute(0, 4, 1, 2, 3)
batch_size = int(sys.argv[2])
train_loader = torch.utils.data.DataLoader(train_cuboids, shuffle=True, batch_size=batch_size, num_workers=4, drop_last=True)

# torch.cuda.set_device(5)
use_cuda = torch.cuda.is_available()
#device = torch.device("cuda:0" if use_cuda else "cpu")
device = torch.device("cuda" if use_cuda else "cpu")
cudnn.benchmark = True

def pretrain(autoencoder, optimizer,
             print_every=100, verbose=False):
    for i, X in enumerate(train_loader):
        # X = Variable(X)
        # X.float().to(device)
        X_var = X
        X_hat = autoencoder(X_var)#autoencoder(X_var.float())
        loss = F.mse_loss(X_hat, X_var.float().to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print('Trn Loss: %.3f' % loss.data)
    
def train(model, optimizer, k):
    gmm_inputs = []
    for i, X in enumerate(train_loader):
        # forward pass and compute loss
        X_var = X#Variable(X)
        clust_ip, d_op  = model(X_var)
        recon_loss = F.mse_loss(d_op, X_var.float().to(device))
        loss = recon_loss 
        for batchs in clust_ip:
          gmm_inputs.append(batchs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Trn Loss: %.3f ',  loss.data)
    return gmm_inputs

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

class KMeansCriterion_no_centroids(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, embeddings, centroids, k):
        """
        Assumes numpy arrays as input
        centroids (shape) = (n_clusters, n_features)
        embeddings = the points to cluster. 
        Returns
        -------
        loss
        cluster_centers? or assignments
        """
        warnings.filterwarnings(action='once')
        kmeans = KMeans(n_clusters=k).fit(embeddings)
        loss = kmeans.inertia_
        centroids = kmeans.cluster_centers_
        return loss, centroids

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
    
class Encoder_d20_2048(nn.Module):
    def __init__(self, dropout=0):
      super().__init__()
      self.conv1 = nn.Conv3d(3, 64, 3, stride=1, padding=1)
      self.conv2 = nn.Conv3d(64, 128, 3, stride=1, padding=1)
      self.conv3 = nn.Conv3d(128, 256, 3, stride=1, padding=1)
      self.conv4 = nn.Conv3d(256, 512, 3, stride=1, padding=1)
      self.pool = nn.AvgPool3d(2, stride=2)
      self.linear = nn.Linear(2048, 10)
      # self.pool2 = nn.AvgPool3d(2, stride=2)
    def forward(self, x):
      x = x.float().to(device)
      # x = x.float()
      x = F.leaky_relu(self.conv1(x))
      x = self.pool(x)
      x = F.leaky_relu(self.conv2(x))
      x = self.pool(x)
      x = F.leaky_relu(self.conv3(x))
      x = self.pool(x)
      x = F.leaky_relu(self.conv4(x))
      embd = self.pool(x)
      clust_in = torch.flatten(embd, 1)
      clust_in = F.leaky_relu(clust_in)
      d = clust_in.reshape(-1, 512, 1, 2, 2)
      return clust_in, d
    

class Decoder_d20_2048(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = Interpolate(scale=2, mode='trilinear')
        self.conv5 = nn.Conv3d(512, 256, 3, stride=1, padding=(2, 1, 1))
        self.conv6 = nn.Conv3d(256, 128, 3, stride=1, padding=(2, 1, 1))
        self.conv7 = nn.Conv3d(128, 64, 3, stride=1, padding=(1, 1, 1))
        self.conv8 = nn.Conv3d(64, 3, 3, stride=1, padding=1)
        self.conv9 = nn.Conv3d(3, 3, 3, stride=(2, 1, 1), padding= 1)
    def forward(self, x):
        if isinstance(x, tuple):
          x = x[1]
        else:
          x = x
        x = self.up(x)
        x = F.leaky_relu(self.conv5(x))
        x = self.up(x)
        x = F.leaky_relu(self.conv6(x))
        x = self.up(x)
        x = F.leaky_relu(self.conv7(x))
        x = self.up(x)
        x = F.leaky_relu(self.conv8(x))
        x = F.leaky_relu(self.conv9(x))
        return x

class Autoencoder(nn.Module):
    
    def __init__(self):
        super(Autoencoder, self).__init__()
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
            # Flatten(),
            nn.LeakyReLU(0.2, inplace=True)
            )
            
        self.decoder = nn.Sequential(
            # UnFlatten(),
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
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Sigmoid()
            )
    
    def encode(self, x):
        x1 = self.encoder(x)#.float().to(device))
        clust_in = torch.flatten(x1, 1)
        clust_in = F.leaky_relu(clust_in)
        d = clust_in.reshape(-1, 512, 1, 2, 2)
        return clust_in, d
    
    def decode(self, x):
        x1 = self.decoder(x)
        
        return x1
    
    def forward(self, x):
        encode_op, d = self.encode(x.float())
        decoder_op = self.decode(d)
        return encode_op, decoder_op
    
        
e = Encoder_d20_2048().cuda()
d = Decoder_d20_2048().cuda()

autoencoder = nn.Sequential(e, d).to(device)
optimizer = optim.Adam(autoencoder.parameters(), amsgrad=True)

model = Autoencoder()
if torch.cuda.device_count() > 1:
  print("Using", torch.cuda.device_count(), "GPUs")
  model = nn.DataParallel(model)
model.to(device)
optimizer = torch.optim.Adam([{'params': model.parameters()}], amsgrad=True)



k, d_size = 9, 2048
criterion = KMeansCriterion_no_centroids()
centroids = None #just initialization

for _ in tqdm(range(num_epochs)):
    gmm_inputs = train(model, optimizer, k)
    

print('Training Completed')

def load_test_data(PIK):
    '''
    for ped1 its both train and test togrthrt
    '''
    with open(PIK, "rb") as f:
        test_images_loaded = pickle.load(f)
        # test_gt_images_loaded = pickle.load(f)
    return test_images_loaded
    # return test_images_loaded, test_gt_images_loaded

def load_test_data_PED1(PIK):
    '''
    for ped1 its both train and test togrthrt
    '''
    with open(PIK, "rb") as f:
        test_images_loaded = pickle.load(f)
        test_gt_images_loaded = pickle.load(f)
    return test_images_loaded, test_gt_images_loaded
    # return test_images_loaded
    

# test_images_PED1, test_images_gt_PED1 = load_test_data_PED1("testing_lists_UCSD1.dat")
test_images = load_test_data("PED2_test_128128.pickle")
test_images_gt = load_test_data("PED2_test_labels_128128.pickle")

# test_images = load_test_data("AvenueTestData.pickle")
# test_images_gt = load_test_data("AvenueTestLabels.pickle")

# test_images = np.load('shanghaiTestImages.npy')
# test_images_gt = np.load('shanghaiLabels.npy')

test_images = make_cuboids(test_images, 20, 32)
label_test_images_np = make_cuboids(test_images_gt, 20, 32) #228 out of 1600 are non-zeros



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
test_inputs = []
for y in test_dataloader:
  clust_pred, d_ignore = model(y)#question: Is the one that is trained or not?
  for i in range(len(clust_pred)):
    test_inputs.append(clust_pred[i].detach().cpu().numpy())

print('Done with testing')



def convert_gmm_inputs_np(gmm_inputs):
  gmm_inputs_np = []
  for i in range(len(gmm_inputs)):
    gmm_inputs_np.append(gmm_inputs[i].detach().cpu().numpy())
  gmm_inputs_np = np.array(gmm_inputs_np)
  return gmm_inputs_np

gmm_inputs_np = convert_gmm_inputs_np(gmm_inputs)

start_time = time.time()
sb = BayesianGaussianMixture(n_components=12, weight_concentration_prior_type='dirichlet_distribution', warm_start=True)  
sb.fit(gmm_inputs_np)
print('Total time:', time.time() - start_time)
# gaussianmixture = GaussianMixture(n_components = 20)
# gaussianmixture.fit(gmm_inputs_np)

print('Stick Breaking Model is Fit!')
data = gmm_inputs_np
print(data.shape)
sb_test_scores = []
count = 0
# for i in range(len(test_inputs)):
start_time = time.time()


with open('anomalies_tracker_PED2.pickle', 'wb') as f:
  pickle.dump(anomaly_tracker, f)


print('Now predicting scores for PED1')
ped1_results = sb_test_scores
test_images_PED1, test_images_gt_PED1 = load_test_data_PED1("testing_lists_UCSD1.dat")


test_images_PED1 = make_cuboids(test_images_PED1, 20, 32)
label_test_images_np_PED1 = make_cuboids(test_images_gt_PED1, 20, 32) 


test_cuboids = []
anomaly_tracker = [] #1 if anomaly is present; 0 if sample is normal
for i in range(len(label_test_images_np_PED1)):
  if np.count_nonzero(label_test_images_np_PED1[i]) > 8192:
    anomaly_tracker.append(1)
    test_cuboids.append(test_images_PED1[i])
  else:
    anomaly_tracker.append(0)
    test_cuboids.append(test_images_PED1[i])
test_cuboids_np = np.array(test_cuboids)
test_cuboids_np = test_cuboids_np.astype('float64')
test_cuboids_np *= 255.0/test_cuboids_np.max()
test_cuboids_t = torch.from_numpy(test_cuboids_np)
test_cuboids_t = test_cuboids_t.permute(0, 4, 1, 2, 3)

    
batch_size = 1
test_dataloader = torch.utils.data.DataLoader(test_cuboids_t, shuffle=False, batch_size=batch_size, num_workers=4, drop_last=True)

print('Test dataloader loaded')
model.eval()
test_inputs = []
for y in test_dataloader:
  clust_pred, d_ignore = model(y)
  for i in range(len(clust_pred)):
    test_inputs.append(clust_pred[i].detach().cpu().numpy())


for sample in test_inputs:
    sb_test_scores.append(sb.score(sample.reshape(1, -1)))
print('Done with testing PED1')

np.save('SB_twoStage_PED1_Unsupervised' +str(num_epochs)+ '.npy', np.array(sb_test_scores))
print('SB results saved')