from torch import nn
import torch
import albumentations as A
import cv2
from tqdm import tqdm
import numpy as np



class Data:
    def __init__(self, data):
        self.data = data
        self.length = len(data)

    def __getitem__(self, i):
        if isinstance(self.data, str): # Путь до картнки
            image = cv2.imread(self.data[i])
        elif isinstance(self.data, torch.Tensor):
            if len(self.data.shape) == 3: # Картинка
                image = to_numpy_image(self.data[i])
            elif len(self.data.shape) == 4: # Батч картинок
                image = to_numpy_image(self.data[i])
        elif isinstance(self.data, list):
            if isinstance(self.data[i], str):
                image = cv2.imread(self.data[i]) # Если путь
            else:
                image = self.data[i]
        else:
            image = self.data[i]
        return image
    
    def __len__(self):
        return self.length
    
    def __iter__(self):
        self.n = 0
        return self
    
    def __next__(self):
        x = self.__getitem__(self.n)
        self.n += 1
        if self.n < self.length:
            return x
        raise StopIteration


class Cutter:
    def __init__(self, data, size=224, step_size=200):
        self.size = size
        self.step_size = step_size
        self.data = data
        self.length = len(self.data)
    
    def __getitem__(self, i):
        image = self.data[i]
        images = []
        for r in range(0, image.shape[0], self.step_size):
            for c in range(0, image.shape[1], self.step_size):
                new_image = image[r:r + self.size, c:c + self.size, :]
                if new_image.shape == (self.size, self.size, 3):
                    images.append([new_image, [c, r, c, r]])
        return images
    
    def __len__(self):
        return self.length

    def __iter__(self):
        self.n = 0
        return self
    
    def __next__(self):
        x = self.__getitem__(self.n)
        self.n += 1
        if self.n < self.length:
            return x
        raise StopIteration

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        model = models.resnet101(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True

        self.model = model
        self.end_layer = nn.Sequential(
            nn.Linear(1000, 128),
            nn.Linear(128, 1)
        )

    def forward(self, X):
        return self.end_layer(self.model(X))
    
    def __call__(self, X):
        return self.forward(X)

class BearCatcher(nn.Module):
    def __init__(self, classifier, detector):
        super().__init__()
        self.classifier = classifier
        self.detector = detector
    
    def forward(self, X):
        detection = []
        normed = normalize(image=X, bboxes=[])['image']
        z = to_torch_image(normed)
        z = z.unsqueeze(0).to(device)
        z = self.classifier(z)
        if not (torch.sigmoid(z) > 0.95):
            return detection
        X = np.array(X, dtype=np.uint8)
        detection = self.detector(X)
        return detection


    def __call__(self, X):
        return self.forward(X)


class Predictor:
    def __init__(self, model, raw_data):
        data = Data(raw_data)
        self.model = model
        self.cutter = Cutter(data)
        self.length = len(self.cutter)

    def __getitem__(self, i):
        self.model.eval()
        images =  self.cutter[i]
        for cutted_image in tqdm(images):
            image, coords = cutted_image
            preds = self.model(image)
            if not isinstance(preds, list):
                output = preds["instances"][preds['instances'].get_fields()['scores'] > 0.9]
                preds = list(list(output.get_fields()['pred_boxes']))
                preds = [list(map(lambda x: x.item(), list(preds[i]))) for i in range(len(preds))]
                coords = np.array(coords)
                coords = np.array([int(coords[1]), int(coords[0]),
                                int(coords[1]), int(coords[0])])
                preds = np.array(preds, dtype=np.int)

            if len(preds) > 0:
                preds = preds + coords
                
                return image, preds
        
    def __len__(self):
        return self.length
    
    def __iter__(self):
        self.n = 0
        return self
    
    def __next__(self):
        if self.n == self.length:
            raise StopIteration
        x = self.__getitem__(self.n)
        if self.n < self.length:
            self.n += 1
            return x
        raise StopIteration


device = 'cpu'

normalize = A.Compose([A.RandomCrop(224, 224), 
                       A.Normalize()], bbox_params={'format':'pascal_voc'})

def to_torch_image(image):
    image = np.moveaxis(image, -1, 0)
    return torch.from_numpy(image)


def to_numpy_image(image):
    image = np.array(image)
    image = np.moveaxis(image, 0, -1)
    return image

def draw_bboxes(image, bboxes, thickness=5):
    for bear in bboxes:
        x1, y1, x2, y2 = bear
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness)
    return image
