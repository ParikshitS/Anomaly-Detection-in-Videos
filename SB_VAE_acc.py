import pickle
import numpy as np
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import math

with open('real_ground_truths_PED2.pickle', 'rb') as f:
  real_ground_truths = pickle.load(f)

print(len(real_ground_truths))
real_np = np.array(real_ground_truths)
unique, counts = np.unique(real_np, return_counts=True)
print(np.asarray((unique, counts)).T)

def f1(tp, fp, fn):
    #2*((precision*recall)/(precision+recall))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2* ((precision*recall)/(precision+recall))
    
def get_scores(filename):
    scores = np.load(filename)
    scores = (scores-min(scores))/(max(scores)-min(scores))
    print(len(scores))
    # with open(filename, 'rb') as f:
    #     scores = pickle.load(f)
    with open('anomalies_tracker_PED2.pickle', 'rb') as f:
        anomaly_tracker = pickle.load(f)
        normal_scores = []
        abnormal_scores = []
        for i in range(len(anomaly_tracker)):
            if anomaly_tracker[i] == 1:
                abnormal_scores.append(scores[i])
            else:
                normal_scores.append(scores[i])
    abnormal_scores = list(abnormal_scores)
    normal_scores = list(normal_scores)
    
    avg_normal = sum(normal_scores) / len(normal_scores)
    avg_abnormal = sum(abnormal_scores) / len(abnormal_scores)
    
    threshold = (avg_normal + avg_abnormal) / 2
    print('Average Normal is {}'.format(avg_normal))
    print('Average Abnormal is {}'.format(avg_abnormal))
    print('The diff b/w Avg Abnormal and normal is {}'.format(avg_abnormal-avg_normal))
    popoulate_predictions = []
    for s in scores:
        if s > threshold:
            popoulate_predictions.extend([1]*20)
        else:
            popoulate_predictions.extend([0]*20)
    
    true_positive_count= 0
    false_positive_count = 0
    true_negative_count = 0
    false_negative_count = 0
    
    print(len(popoulate_predictions), len(real_ground_truths), len(scores))
    for i in range(len(popoulate_predictions)):
        if real_ground_truths[i] == 1 and popoulate_predictions[i] == 1:
            true_positive_count += 1
        if real_ground_truths[i] == 1 and popoulate_predictions[i] == 0:
            false_negative_count += 1
        if real_ground_truths[i] == 0 and popoulate_predictions[i] == 1:
            false_positive_count += 1
        if real_ground_truths[i] == 0 and popoulate_predictions[i] == 0:
            true_negative_count += 1
    
    
    tps = []
    fps = []
    # f1_score = []
    th = math.floor(avg_normal) #math.floor(min(normal_scores))
    th = math.floor(min(abnormal_scores))  # avg_abnormal
    while th < math.floor(max(normal_scores)):
        # print(th)
        true_positive_count= 0
        false_positive_count = 0
        true_negative_count = 0
        false_negative_count = 0
        
        popoulate_predictions = []
        for s in scores:
            if s < th:
                popoulate_predictions.extend([1]*20)
            else:
                popoulate_predictions.extend([0]*20)
            
        for i in range(len(popoulate_predictions)):
            if (real_ground_truths[i] == 1) and (popoulate_predictions[i] == 1):
                true_positive_count += 1
            if (real_ground_truths[i] == 1) and (popoulate_predictions[i] == 0):
                false_negative_count += 1
            if (real_ground_truths[i] == 0) and (popoulate_predictions[i] == 0):
                true_negative_count += 1       
            if(real_ground_truths[i] == 0) and (popoulate_predictions[i] == 1):
                false_positive_count += 1
        
        # print(true_positive_count, false_negative_count, false_positive_count, true_negative_count)
        true_postive_rate = true_positive_count / (true_positive_count + false_negative_count)
        false_positive_rate = false_positive_count / (false_positive_count + true_negative_count)
        if false_positive_rate != 0:
            tps.append(true_postive_rate)
            fps.append(false_positive_rate)
        
        # f1_score.append(f1(true_positive_count, false_positive_count, false_negative_count))
        th += 0.05
    return fps, tps

print('here')
sb_fps, sb_tps =  get_scores('SB_twoStage_PED2_Unsupervised1.npy')


plt.plot(sb_fps, sb_tps, linestyle='dashdot', color='blue', label='Stick Breaking VAE')
plt.legend(loc="best")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve on Stick Breaking VAE on PED1')
plt.show()
print(auc(np.array(sb_fps), np.array(sb_tps)))