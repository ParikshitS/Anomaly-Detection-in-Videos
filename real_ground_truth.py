import pickle
import numpy as np

def load_test_data(PIK):
    '''
    for ped1 its both train and test togrthrt
    '''
    with open(PIK, "rb") as f:
        test_images_loaded = pickle.load(f)
        test_gt_images_loaded = pickle.load(f)
    return test_gt_images_loaded

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

test_images_gt = np.load('shanghaiLabels.npy')
cuboids = make_cuboids(test_images_gt, 20, 32)

real_ground_truths = []
for c in range(len(cuboids)):
    for f in range(20):
        if np.count_nonzero(cuboids[c][f]) > 409:
            real_ground_truths.append(1)
        else:
            real_ground_truths.append(0)

with open('real_ground_truths_Shanghai.pickle', 'wb') as f:
  pickle.dump(real_ground_truths, f)