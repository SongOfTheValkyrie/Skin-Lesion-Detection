import torch
from torchvision.models import mobilenet_v3_small
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights
from torchvision.models.mobilenetv3 import MobileNetV3 
import csv
from PIL import Image
from torchvision import transforms
import os
import numpy as np
from sklearn.model_selection import train_test_split
import torchsummary
import math
from time import time
from datetime import timedelta
from ham10k_test import ham10k_test

# !!! change useCuda to False to run un CPU (if program crashes) !!!
useCuda = True

if useCuda:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

imgs_paths = ['HAM10000/HAM10000_images/']
labels_path = 'HAM10000/HAM10000_metadata'
#hamtest = ham10k_test(10)

batch_size = 32
#diagnostic categories
one_hot_dict = {
    'nv' : torch.Tensor([1, 0, 0, 0, 0, 0, 0]), #melanocytic nevi
    'mel' : torch.Tensor([0, 1, 0, 0, 0, 0, 0]), #melanoma
    'bkl' : torch.Tensor([0, 0, 1, 0, 0, 0, 0]), #solar lentigines / seborrheic keratoses and lichen-planus like keratoses
    'bcc' : torch.Tensor([0, 0, 0, 1, 0, 0, 0]), # basal cell carcinoma
    'akiec' : torch.Tensor([0, 0, 0, 0, 1, 0, 0]), #Actinic keratoses and intraepithelial carcinoma / Bowen's disease
    'df' : torch.Tensor([0, 0, 0, 0, 0, 1, 0]), #dermatofibroma
    'vasc' : torch.Tensor([0, 0, 0, 0, 0, 0, 1]) # vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage
}
n_epochs = 20

def load_trainset(imgs_paths, labels_path, train_percentage):
    with open(labels_path, 'r') as labels_file:
        img_to_label = {row['image_id'] : row['dx'] for row in csv.DictReader(labels_file)}
    
    imgs = []
    labels = []
    for imgs_path in imgs_paths:
        for file in os.listdir(imgs_path):
            imgs.append(imgs_path + file)
            labels.append(img_to_label[file.split('.')[0]])
    
    train_num = round(len(imgs) * train_percentage)
    return np.array(imgs[: train_num]), np.array(labels[: train_num])

def balance_trainset(imgs, labels):
    class_imgs = {}
    for img, label in zip(imgs, labels):
        if label not in class_imgs:
            class_imgs[label] = [img]
        else:
            class_imgs[label].append(img)
    
    for label in class_imgs:
        class_imgs[label] = np.array(class_imgs[label])
    
    majority_class = max(class_imgs, key = lambda label : len(class_imgs[label]))
    majority_class_num = len(class_imgs[majority_class])
    
    for label in class_imgs:
        if label != majority_class:
            chosen_indices = np.random.choice(np.arange(len(class_imgs[label])), majority_class_num - len(class_imgs[label]))
            imgs = np.concatenate((imgs, class_imgs[label][chosen_indices]))
            labels = np.concatenate((labels, np.repeat(label, majority_class_num - len(class_imgs[label]))))
    
    return imgs, labels

def img_batches(imgs, labels):
    batch_imgs = []
    batch_labels = []
    for img_path, label in zip(imgs, labels):
        with Image.open(img_path) as pil_img:
            img = transforms.ToTensor()(pil_img)
            if useCuda:
                img = img.cuda()
        batch_imgs.append(img[None, :])
        batch_labels.append(one_hot_dict[label][None, :])
        if len(batch_imgs) == batch_size:
            yield torch.cat(batch_imgs, 0), torch.cat(batch_labels, 0)
                
            batch_imgs = []
            batch_labels = []
            
    if len(batch_imgs) != 0:
        yield torch.cat(batch_imgs, 0), torch.cat(batch_labels, 0)

def forward(model : MobileNetV3, x):
    x = model.features(x)
    
    x = model.avgpool(x)
    z = torch.flatten(x, 1)

    x = model.classifier(z)
    return x, z

def save_tensor(x, file_path):
    if len(x.shape) == 1:
        x = x[None, :]
    
    with open(file_path, "w") as f:
        f.write(f"{x.shape[0]}\n{x.shape[1]}\n")
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                f.write(f"{x[i, j]}\n")

print('Loading model...')
model = mobilenet_v3_small(weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False

model.classifier = torch.nn.Sequential(
    torch.nn.Linear(576, 1024),
    torch.nn.Hardswish(),
    torch.nn.Dropout(p = 0.2),
    torch.nn.Linear(1024, 7),
    torch.nn.Softmax()
)

torchsummary.summary(model, input_size=(3, 450, 600), batch_size=1)

optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 20, 30], gamma=0.1)
criterion = torch.nn.CrossEntropyLoss()

# prepare mahalanobis distance params
mean_feature_maps = torch.zeros((7, 576))
mean_feature_map_0 = torch.zeros(576)
Nk = torch.zeros(7)
covar = torch.zeros((576, 576))
covar_0 = torch.zeros((576, 576))

# load train set
imgs, labels = load_trainset(imgs_paths, labels_path, 0.9)


with torch.no_grad():
    # Calculate mean feature maps
    print("Calculating mean feature maps...")
    for batch, label in img_batches(imgs, labels):
        out, z = forward(model, batch)
        for feature_map, label_one_hot in zip(z, label):
            label_num = torch.argmax(label_one_hot)
            mean_feature_maps[label_num] += feature_map
            Nk[label_num] += 1
    
    mean_feature_map_0 = torch.sum(mean_feature_maps, dim = 0) / torch.sum(Nk)
    mean_feature_maps = (mean_feature_maps.T / Nk).T
    torch.save(mean_feature_map_0, "ood_params/mean_feature_map_0.pt")
    save_tensor(mean_feature_map_0, "ood_params/mean_feature_map_0.matrix")
    torch.save(mean_feature_maps, "ood_params/mean_feature_maps.pt")
    save_tensor(mean_feature_maps, "ood_params/mean_feature_maps.matrix")
    
    # Calculate covariance matrices
    print("Calculating covariance matrices")
    for batch, label in img_batches(imgs, labels):
        out, z = forward(model, batch)
        for feature_map, label_one_hot in zip(z, label):
            label_num = torch.argmax(label_one_hot)
            feature_map_adjusted_0 = (feature_map - mean_feature_map_0)[None, :]
            feature_map_adjusted = (feature_map - mean_feature_maps[label_num])[None, :]
            covar_0 += feature_map_adjusted_0.T @ feature_map_adjusted_0
            covar += feature_map_adjusted.T @ feature_map_adjusted
    
    covar_0 = covar_0 / torch.sum(Nk)
    covar = covar / torch.sum(Nk)
    torch.save(torch.linalg.inv(covar_0), "ood_params/covar_0_inverse.pt")
    save_tensor(torch.linalg.inv(covar_0), "ood_params/covar_0_inverse.matrix")
    torch.save(torch.linalg.inv(covar), "ood_params/covar_inverse.pt")
    save_tensor(torch.linalg.inv(covar), "ood_params/covar_inverse.matrix")

# oversample minority classes to make it balanced
imgs, labels = balance_trainset(imgs, labels)

print("Training model...")
for epoch in range(1, n_epochs + 1):
    train_losses = []
    indices = np.arange(imgs.shape[0])
    np.random.shuffle(indices)
    imgs = imgs[indices]
    labels = labels[indices]
    
    epoch_start = time()
    for batch, label in img_batches(imgs, labels):
        model.zero_grad()
        # Get prediction and feature map
        out, z = forward(model, batch)
        # Calculate loss
        loss = criterion(out, label)
        # Perform backprop
        loss.backward()
        # Update weights
        optimizer.step()
        train_losses.append(loss.item())
            
    scheduler.step()
    epoch_end = time()
    
    print(f'Epoch: {epoch}, Loss: {np.mean(train_losses)}, Elapsed time: {timedelta(seconds = epoch_end - epoch_start)}')

torch.save(model.state_dict(), "trained_models/ham10k_trained.pth")


#include test
#hamtest.test()
