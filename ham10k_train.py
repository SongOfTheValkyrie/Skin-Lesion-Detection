import torch
from models import Mobilenet_v3_large
import csv
from PIL import Image
from torchvision import transforms
import os
import numpy as np
from sklearn.model_selection import train_test_split
import torchsummary
from ham10k_test import ham10k_test

#torch.set_default_tensor_type(torch.cuda.FloatTensor)
imgs_paths = ['HAM10000/HAM10000_images/']
labels_path = 'HAM10000/HAM10000_metadata'
hamtest=ham10k_test()

batch_size=8
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
n_epochs = 5

def img_batches(imgs_paths, labels):
    
    with open(labels_path, 'r') as labels_file:
        labels = {row['image_id'] : row['dx'] for row in csv.DictReader(labels_file)}
    
    batch_imgs = []
    batch_labels = []
    for imgs_path in imgs_paths:
        for file in os.listdir(imgs_path):
            with Image.open(imgs_path + file) as pil_img:
                img = transforms.ToTensor()(pil_img)
                #print(img.shape)
            batch_imgs.append(img[None, :])
            batch_labels.append(one_hot_dict[labels[file.split('.')[0]]][None, :])
            """if len(batch_imgs) =:
                yield torch.cat(batch_imgs, 0), torch.cat(batch_labels, 0)
                batch_imgs = []
                batch_labels = []"""
    return batch_imgs,batch_labels

print('Loading model...')
model = Mobilenet_v3_large()
try:
    model.load_state_dict(torch.load('trained_models/Mobilenet_v3_large_1.0_fortest.pth'))
except:
    model.load_state_dict(torch.load('trained_models/Mobilenet_v3_large_1.0_fortest.pth',map_location='cpu'))

torchsummary.summary(model, input_size=(3, 450, 600), device='cpu',batch_size=1)

for param in model.parameters():
    param.requires_grad = False

model.conv_4 = torch.nn.Sequential(
    torch.nn.Conv2d(1280, 7, 1),
    torch.nn.Softmax()
)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 20, 30], gamma=0.1)

with open(labels_path, 'r') as labels_file:
        labels_dict = {row['image_id'] : row['dx'] for row in csv.DictReader(labels_file)}

#separate test and train sets. 
batch, labels=img_batches(imgs_paths, labels_dict) 
percent_train=round(len(batch)/100*90)
batch_train=batch[0:percent_train]
batch_test=batch[percent_train:]

label_train=labels[0:percent_train]
label_test=labels[percent_train:]


for epoch in range(1, n_epochs + 1):
    train_losses = []
    for count in range(1,len(batch_train)+1):
        if count%batch_size==0:
            model.zero_grad()
            #, torch.cat(batch_labels, 0)
            _batch=batch_train[count-batch_size:count]
            _label=label_train[count-batch_size:count]
            underbatch=torch.cat(_batch, 0)
            underlabel=torch.cat(_label, 0)
            out = model.forward(underbatch)
            #print(out.shape)
            #print(labels.shape)
            loss = criterion(out, torch.max(underlabel, 1)[1])
            print(loss.item())
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
    scheduler.step()
    
    print(f'Epoch: {epoch}, Loss: {np.mean(train_losses)}')

torch.save(model.state_dict(), "/kaggle/working/ham10k_trained.pth")

#include test
hamtest.test(batch_test,label_test)
