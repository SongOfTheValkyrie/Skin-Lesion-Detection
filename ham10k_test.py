import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from torchvision.models import mobilenet_v3_large
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights 
from torchvision.models.mobilenetv3 import MobileNetV3
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
imgs_paths = ['HAM10000/HAM10000_images/']

class ham10k_test:
    def __init__(self, num_mc_passes):
        self.mean_feature_map_0 = torch.load("ood_params/mean_feature_map_0.pt")
        self.mean_feature_maps = torch.load("ood_params/mean_feature_maps.pt")
        self.covar_inverse = torch.load("ood_params/covar_inverse.pt")
        self.covar_0_inverse = torch.load("ood_params/covar_0_inverse.pt")
        
        self.num_mc_passes = num_mc_passes
    
    one_hot_dict = {
        'nv' : 0, 
        'mel' : 1,
        'bkl' : 2,
        'bcc' : 3,
        'akiec' : 4,
        'df' : 5,
        'vasc' : 6
    }
    
    conf = torch.zeros(7, 7)

    #test directly after training
    def test(self):
        model, imgs, labels = self.load_model_and_images()
        print(f"Testing on last {len(imgs)} images")

        confidences=[]
        i = 0
        for img, label in zip(imgs, labels):
            golden, predicted, rmd_confidence = self.predict(img, label, model)
            
            predicted = torch.argmax(predicted).item()
            actual = self.one_hot_dict[golden]
            confidences.append(rmd_confidence)
            
            self.conf[actual, predicted] += 1
            
            #print(f"Expected: {golden:5}, Predicted: {predicted_var[i]:5}, Certainty: {torch.max(predicted):.2f}, In-distribution confidence: {rmd_confidence:.2f}")
            if (i + 1) % 10 == 0:
                acc = torch.sum(torch.diag(self.conf)) / torch.sum(self.conf)
                print(f"First {i + 1} images: Acc={acc * 100:.2f}% Avg_confidence={np.mean(confidences)}")
            i += 1
    
        precision = np.zeros(7)
        recall = np.zeros(7)
    
        for i in range(7):
            if self.conf[i, i] == 0:
                precision[i] = 0
                recall[i] = 0
            else:
                precision[i] = self.conf[i, i] / torch.sum(self.conf[:, i])
                recall[i] = self.conf[i, i] / torch.sum(self.conf[i, :])
        
        for i in self.one_hot_dict:
            ind = self.one_hot_dict[i]
            print(f"{i}: Precision {precision[ind]:.2f}, Recall {recall[ind]:.2f}")


    def load_model_and_images(self):

        model = mobilenet_v3_small(weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(576, 1024),
            torch.nn.Hardswish(inplace = True),
            torch.nn.Dropout(p = 0.2, inplace = True),
            torch.nn.Linear(1024, 7),
            torch.nn.Softmax()
        )
        model.load_state_dict(torch.load('trained_models/ham10k_trained.pth'))
        model.eval()
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

        #load all HAMM images
        with open(labels_path, 'r') as labels_file:
            labels = {row['image_id'] : row['dx'] for row in csv.DictReader(labels_file)}
            
        all_imgs = []
        all_labels = []
        for imgs_path in imgs_paths:
            all_imgs = all_imgs + [imgs_path + file for file in os.listdir(imgs_path)]
            all_labels = all_labels + [labels[file.split('.')[0]] for file in os.listdir(imgs_path)]             

        percent_train = round(len(all_imgs) * 0.8)
        percent_dev = round((len(all_imgs) - percent_train) / 2)
        
        all_imgs = all_imgs[percent_train + percent_dev :]
        all_labels = all_labels[percent_train + percent_dev :]
        return model, all_imgs, all_labels
    
    def mahalanobis_distance(self, feature_map, mean_feature_map, covar_inverse):
        adjusted_feature_map = (feature_map - mean_feature_map)[None, :]
        return (adjusted_feature_map @ covar_inverse @ adjusted_feature_map.T)[0][0].item()
    
    
    def rmd_confidence(self, feature_map, mean_feature_maps, mean_feature_map_0, covar_inverse, covar_0_inverse):
        md_0 = self.mahalanobis_distance(feature_map, mean_feature_map_0, covar_0_inverse)
        
        rmd = [(self.mahalanobis_distance(feature_map, mean_feature_map, covar_inverse) - md_0) for mean_feature_map in mean_feature_maps]
        
        return -min(rmd)
    
    def forward(self, model : MobileNetV3, x):
        x = model.features(x)
    
        x = model.avgpool(x)
        z = torch.flatten(x, 1)

        x = model.classifier(z)
        return x, z
    
    def predict(self, img_path, golden, model):
        #random_img_path = img_path[random.randint(0, len(img_path) - 1)]

        with torch.no_grad():
            with Image.open(img_path) as pil_img:
                img = transforms.ToTensor()(pil_img)
                if useCuda:
                    img = img.cuda()
            
            predictions = []
            for i in range(self.num_mc_passes):
                predicted, feature_map = self.forward(model, img[None, :])
                predictions.append(predicted)
            predictions = torch.cat(predictions)
            return golden, torch.mean(predictions, dim = 0), self.rmd_confidence(feature_map[0], self.mean_feature_maps, self.mean_feature_map_0, self.covar_inverse, self.covar_0_inverse)

if __name__ == '__main__':
    hamtest = ham10k_test(10)
    hamtest.test()
