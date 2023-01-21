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
imgs_paths = 'HAM10000/HAM10000_images/'

class ham10k_test:
    def __init__(self, num_mc_passes):
        self.mean_feature_map_0 = torch.load("ood_params/mean_feature_map_0.pt")
        self.mean_feature_maps = torch.load("ood_params/mean_feature_maps.pt")
        self.covar_inverse = torch.load("ood_params/covar_inverse.pt")
        self.covar_0_inverse = torch.load("ood_params/covar_0_inverse.pt")
        
        self.num_mc_passes = num_mc_passes
    
    one_hot_dict = {
        0:'nv', 
        1:'mel',
        2:'bkl',
        3:'bcc',
        4:'akiec',
        5:'df',
        6:'vasc'
    }

    #test directly after training
    def test(self):
        model, imgs, label = self.load_model_and_images()
        print(f"Testing on last {len(imgs)} images")

        predicted_var=[]
        actual=[]
        preds=[]
        confidences=[]

        for i, img in enumerate(imgs):
            golden, predicted, rmd_confidence = self.predict(img, label, model)
            predicted_var.append(self.one_hot_dict[torch.argmax(predicted).item()])
            actual.append(golden)
            preds.append(predicted)
            confidences.append(rmd_confidence)
            #print(f"Expected: {golden:5}, Predicted: {predicted_var[i]:5}, Certainty: {torch.max(predicted):.2f}, In-distribution confidence: {rmd_confidence:.2f}")
            if (i + 1) % 10 == 0:
                acc = np.sum(np.array(actual) == np.array(predicted_var)) / len(actual)
                print(f"First {i + 1} images: Acc={acc * 100:.2f} Avg_confidence={np.mean(confidences)}")


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
        all_imgs = [file for file in os.listdir(imgs_paths)]
        with open(labels_path, 'r') as labels_file:
            labels = {row['image_id'] : row['dx'] for row in csv.DictReader(labels_file)}

        percent_train = round(len(all_imgs) * 0.9)
        
        all_imgs = all_imgs[percent_train :]
        return model, all_imgs,labels
    
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
    
    def predict(self, img_path, label, model):
        #random_img_path = img_path[random.randint(0, len(img_path) - 1)]
        golden = label[img_path.split('.')[0]]

        with torch.no_grad():
            with Image.open(imgs_paths + img_path) as pil_img:
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
