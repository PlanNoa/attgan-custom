import torch
from slimcnn.slimnet import SlimNet
from torchvision import transforms
from PIL import Image
import numpy as np

class slimcnn:

    def __init__(self):

        self.device = torch.device('cpu')
        self.transform = transforms.Compose([
                                      transforms.Resize((178,218)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ])
        self.model = SlimNet.load_pretrained('slimcnn/models/celeba_20.pth').to(self.device)


    def get_attr(self, PATH):

        with open(PATH, 'rb') as f:
            x = self.transform(Image.open(f)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            self.model.eval()
            logits = self.model(x)
            r = list(map(lambda x: 1 if x > 0.5 else -1, np.array(torch.sigmoid(logits))[0]))

        filter = [5,6,9,10,12,13,16,21,22,23,25,27,40]
        attrs = []
        for idx, attr in enumerate(r):
            if idx + 1 in filter:
                attrs.append(attr)

        return attrs

