import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from torchvision.models import mobilenet_v3_large
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights 
import csv
from PIL import Image
from torchvision import transforms
import numpy as np
import os

# !!! change useCuda to False to run un CPU (if program crashes) !!!
useCuda = True

if useCuda:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

labels_path = 'HAM10000/HAM10000_metadata'
imgs_paths = 'HAM10000/HAM10000_images/'

class ham10k_test:
    one_hot_dict = {
        0:'nv', 
        1:'mel',
        2:'bkl',
        4:'bcc',
        5:'akiec',
        6:'df',
        7:'vasc'
    }

    #test directly after training
    def test(self):
        model, imgs, label = self.load_model_and_images()
        print(f"Testing on last {len(imgs)} images")

        predicted_var=[]
        actual=[]
        preds=[]

        for i, img in enumerate(imgs):
            golden,predicted = self.predict(img, label, model)
            predicted_var.append(self.one_hot_dict[predicted])
            actual.append(golden)
            preds.append(predicted)
            acc = np.sum(np.array(actual) == np.array(predicted_var)) / len(actual)
            if (i + 1) % 10 == 0:
                print(f"Accuracy for first {i + 1} test images: {acc * 100:.2f}%")


    def load_model_and_images(self):

        model = mobilenet_v3_small(weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        model.classifier[-1] = torch.nn.Sequential(
            torch.nn.Linear(1024, 7),
            torch.nn.Softmax()
        )
        model.load_state_dict(torch.load('trained_models/ham10k_trained.pth'))

        #load all HAMM images
        all_imgs = [file for file in os.listdir(imgs_paths)]
        with open(labels_path, 'r') as labels_file:
            labels = {row['image_id'] : row['dx'] for row in csv.DictReader(labels_file)}

        percent_train = round(len(all_imgs) * 0.9)
        
        all_imgs = all_imgs[percent_train :]
        return model, all_imgs,labels
    
    def predict(self,img,label,model):
        #random_img_path = img_path[random.randint(0, len(img_path) - 1)]
        golden = label[img.split('.')[0]]

        with torch.no_grad():
            with Image.open(imgs_paths + img) as pil_img:
                img = transforms.ToTensor()(pil_img)
                if useCuda:
                    img = img.cuda()
            predicted = model.forward(img[None, :])
            return golden, torch.argmax(predicted).item()
    
    def MC_Dropout(self,model):
        pass

if __name__ == '__main__':
    hamtest = ham10k_test()
    hamtest.test()