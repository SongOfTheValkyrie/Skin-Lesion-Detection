import torch
from models import Mobilenet_v3_large
import csv
from PIL import Image
from torchvision import transforms
import numpy as np
import os

#when I use this, everything crashes
#torch.set_default_tensor_type(torch.cuda.FloatTensor)

labels_path = 'HAM10000/HAM10000_metadata'
imgs_paths = 'HAM10000/HAM10000_images/'

class ham10k_test:

    #test directly after training
    def test(self,imgs,labels):
        model = Mobilenet_v3_large()
        model.conv_4 = torch.nn.Sequential(
            torch.nn.Conv2d(1280, 7, 1),
            torch.nn.Softmax()
        )
        model.load_state_dict(torch.load('trained_models/ham10k_trained.pth'))

        for number in range(len(imgs)):
            golden = labels[number]

            with torch.no_grad():
                predicted = model.forward(imgs[number][None, :])
                print(golden)
                print(predicted)


    def load_model_and_images(self):

        model = Mobilenet_v3_large()
        model.conv_4 = torch.nn.Sequential(
            torch.nn.Conv2d(1280, 7, 1),
            torch.nn.Softmax()
        )
        try:
            model.load_state_dict(torch.load('trained_models/ham10k_trained.pth'))
        #if no CUDA available
        except:
             model.load_state_dict(torch.load('trained_models/ham10k_trained.pth',map_location='cpu'))

        #load all HAMM images
        all_imgs = [file for file in os.listdir(imgs_paths)]
        with open(labels_path, 'r') as labels_file:
            labels = {row['image_id'] : row['dx'] for row in csv.DictReader(labels_file)}

        percent_train=round(len(all_imgs)/100*10)
        
        all_imgs=all_imgs[percent_train:]
        return model, all_imgs,labels
    
    def predict(self,img,label,model):
        #random_img_path = img_path[random.randint(0, len(img_path) - 1)]
        golden = label[img.split('.')[0]]

        with torch.no_grad():
            with Image.open(imgs_paths + img) as pil_img:
                img = transforms.ToTensor()(pil_img)
            predicted = model.forward(img[None, :])
            return golden,np.argmax(predicted)


one_hot_dict = {
    0:'nv', 
    1:'mel',
    2:'bkl',
    4:'bcc',
    5:'akiec',
    6:'df',
    7:'vasc'
}

hamtest=ham10k_test()
model,imgs,label=hamtest.load_model_and_images()

predicted_var=[]
actual=[]
preds=[]
#monte carlo dropout
for i in range(3):
    golden,predicted=(hamtest.predict(imgs[i],label,model))
    predicted_var.append(one_hot_dict[int(predicted)])
    actual.append(golden)
    preds.append(int(predicted))
    print(golden,one_hot_dict[int(predicted)])
acc=np.mean(actual==predicted_var)
print("Accuracy:",acc*100)


