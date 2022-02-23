import numpy as np
import os
from tifffile import imsave, imread

# get the sum of intersection and union over all chips
intersection = 0
union = 0

preds = os.listdir("./codeexecution/predictions")
file_pairs = []
for pred in preds:
    real = "./data/data/train_labels/" + pred
    assert os.path.isfile(real)
    pred = "./codeexecution/predictions/"+pred
    file_pairs.append((pred, real))
for pred, actual in file_pairs:
    actual = imread(actual)
    pred = imread(pred)
    #print(actual.max(), actual.std(), actual.min(), pred.max(), pred.min())
    intersection += np.logical_and(actual, pred).sum()

    union += np.logical_or(actual, pred).sum()

# calculate the score across all chips
iou = intersection / union 
print(iou)