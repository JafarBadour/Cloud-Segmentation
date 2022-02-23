from dataloader import DataLoader
import torch

import matplotlib.pyplot as plt


def plot_1k_step(model, ROOT_DIRECTORY, INPUT_IMAGES_DIRECTORY):
    dataloader = DataLoader(ROOT_DIRECTORY, INPUT_IMAGES_DIRECTORY, mode="TRAIN")
    imgs_n = 3
    data = [dataloader.random() for x in range(imgs_n)]
    
    fig, axes = plt.subplots(2 , imgs_n)
    for cnt, (x, y_true) in enumerate(data):
        y_pred = model(numpy2tensor(x))
        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.reshape(512,512)
        y_pred = y_pred.reshape(512,512)
        ax = axes[0][cnt]
        ax.imshow(y_true)
        ax.set_title("Real")
        ax = axes[1][cnt]
        ax.imshow(y_pred)
        ax.set_title("Pred")
    fig.suptitle("Results on random sample")
    #fig.tight_layout()
    plt.show()
        
def numpy2tensor(x, device='cuda:0',dtype=torch.FloatTensor):
    x = x.astype(float)
    x = torch.from_numpy(x)
    x = x.type(dtype).to(device)
    return x

def my_loss(output, target):
    loss = (output - target)**2
    return loss