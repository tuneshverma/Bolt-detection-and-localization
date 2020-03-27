import os
import time
import cv2
import argparse
from tqdm import tqdm
import torchvision.models as models
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

from PIL import Image

from segment_detection import radiuswithcenter, segment_detection


class DEFECT_DETECTOR:

    def __init__(self, model_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet34()
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.num_ftrs, 2)
        )
        self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.loader = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def image_loader(self, image_name):
        # image = Image.open(image_name)
        image = self.loader(image_name).float()
        image = Variable(image, requires_grad=True)
        image = image.unsqueeze(0)
        return image.to(self.device)

    def defect_detector(self, image):
        image = cv2.imread(image)
        image_resize = cv2.resize(image, (640, 640))
        list_of_contours, dictionary = segment_detection(image_resize)
        prop_coor = {}
        q = 0
        for pos in range(len(list_of_contours)):
            c_x, c_y, r = radiuswithcenter(image_resize, dictionary, list_of_contours, pos, 1900, draw=False)
            if c_x:
                if c_x - 75 < 0:
                    x1 = 0
                else:
                    x1 = c_x - 75
                if c_x + 75 > 640:
                    x2 = 640
                else:
                    x2 = c_x + 75
                if c_y - 75 < 0:
                    y1 = 0
                else:
                    y1 = c_y - 75
                if c_y + 75 > 640:
                    y2 = 640
                else:
                    y2 = c_y + 75

                cropped_image = image_resize[y1:y2, x1:x2]
                # cv2.imwrite('./cropped_images/' + str(pos) + '.png', cropped_image)
                cropped_image = Image.fromarray(cropped_image)
                tranformed_image = self.image_loader(cropped_image)
                with torch.no_grad():
                    prediction = self.model(tranformed_image)
                    # print(prediction)
                    _, preds = torch.max(prediction, 1)
                    # print(preds)
                    preds = preds.cpu().numpy()
                    # print(preds)
                    if preds[0] == 0:
                        q = 1
                    prediction = prediction.cpu().numpy()
                    # print(prediction)
                    prop_coor[prediction[0][0]] = [x1, x2, y1, y2]
        # print(prop_coor)
        # print(max(prop_coor.keys()))
        # print(q)
        if q == 1:
            status = 'withbolt'
            x1, x2, y1, y2 = prop_coor[max(prop_coor.keys())]
            cv2.rectangle(image_resize, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow('with_bolt', image_resize)
            cv2.waitKey(0)
            return status, (x1, x2, y1, y2)
        else:
            status = 'withoutbolt'
            cv2.imshow('without_bolt', image_resize)
            cv2.waitKey(0)
            return status, None


# df = DEFECT_DETECTOR('./saved_model/resnet34.pth')
# time1 = time.time()
# c = 0
# for image in tqdm(os.listdir('./main_data/bad_image')):
#     pred_ = df.defect_detector('./main_data/bad_image/' + image)
#     if pred_ == 1:
#         c += 1
# print(c)
# print(time.time() - time1)


parser = argparse.ArgumentParser()
parser.add_argument('--imagepath', help='image path')
args = parser.parse_args()
df = DEFECT_DETECTOR('./saved_model/resnet34.pth')
status, coor = df.defect_detector(args.imagepath)
print(status)
print(coor)
