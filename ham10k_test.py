import torch
from models import Mobilenet_v3_large
import csv
from PIL import Image
from torchvision import transforms
import os
import numpy as np
import torchsummary
import random

torch.set_default_tensor_type(torch.cuda.FloatTensor)

labels_path = 'HAM10000/HAM10000_metadata'

with open(labels_path, 'r') as labels_file:
        labels_dict = {row['image_id'] : row['dx'] for row in csv.DictReader(labels_file)}

model = Mobilenet_v3_large()
model.conv_4 = torch.nn.Sequential(
    torch.nn.Conv2d(1280, 7, 1),
    torch.nn.Softmax()
)
model.load_state_dict(torch.load('trained_models/ham10k_trained.pth'))

imgs_path = 'HAM10000/HAM10000_images/'
all_imgs = [file for file in os.listdir(imgs_path)]

random_img_path = all_imgs[random.randint(0, len(all_imgs) - 1)]

with Image.open(imgs_path + random_img_path) as pil_img:
    img = transforms.ToTensor()(pil_img).cuda()

golden = labels_dict[random_img_path.split('.')[0]]

with torch.no_grad():
    predicted = model.forward(img[None, :])
    print(golden)
    print(predicted)
    