import torch
from models import Mobilenet_v3_large
import csv
from PIL import Image
from torchvision import transforms
import os
import numpy as np
import torchsummary

torch.set_default_tensor_type(torch.cuda.FloatTensor)

imgs_paths = ['HAM10000/HAM10000_images/']
labels_path = 'HAM10000/HAM10000_metadata'
batch_size = 8
one_hot_dict = {
    'nv' : torch.Tensor([1, 0, 0, 0, 0, 0, 0]),
    'mel' : torch.Tensor([0, 1, 0, 0, 0, 0, 0]),
    'bkl' : torch.Tensor([0, 0, 1, 0, 0, 0, 0]),
    'bcc' : torch.Tensor([0, 0, 0, 1, 0, 0, 0]),
    'akiec' : torch.Tensor([0, 0, 0, 0, 1, 0, 0]),
    'df' : torch.Tensor([0, 0, 0, 0, 0, 1, 0]),
    'vasc' : torch.Tensor([0, 0, 0, 0, 0, 0, 1])
}
n_epochs = 10

def img_batches(imgs_paths, labels, batch_size):
    
    with open(labels_path, 'r') as labels_file:
        labels = {row['image_id'] : row['dx'] for row in csv.DictReader(labels_file)}
    
    batch_imgs = []
    batch_labels = []
    for imgs_path in imgs_paths:
        for file in os.listdir(imgs_path):
            with Image.open(imgs_path + file) as pil_img:
                img = transforms.ToTensor()(pil_img)
                print(img.shape)
            batch_imgs.append(img[None, :])
            batch_labels.append(one_hot_dict[labels[file.split('.')[0]]][None, :])
            if len(batch_imgs) == batch_size:
                yield torch.cat(batch_imgs, 0), torch.cat(batch_labels, 0)
                batch_imgs = []
                batch_labels = []

print('Loading model...')
model = Mobilenet_v3_large()
model.load_state_dict(torch.load('trained_models/Mobilenet_v3_large_1.0_fortest.pth'))

torchsummary.summary(model, input_size=(3, 450, 600), device='cpu', batch_size=1)

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

for epoch in range(1, n_epochs + 1):
    train_losses = []
    for batch, labels in img_batches(imgs_paths, labels_dict, batch_size):
        model.zero_grad()
        out = model.forward(batch)
        print(out.shape)
        print(labels.shape)
        loss = criterion(out, labels)
        print(loss.item())
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    scheduler.step()
    
    print(f'Epoch: {epoch}, Loss: {np.mean(train_losses)}')

torch.save(model.state_dict(), "/kaggle/working/ham10k_trained.pth")
