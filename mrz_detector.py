import pickle

import cv2
import numpy as np
import torch
import torchvision
import pytesseract
from torch import nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.backbone = torchvision.models.resnet34(pretrained=True)
        self.fc = nn.Linear(1000, 16)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

    def predict(self, image):
        o_h, o_w = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))
        image = image.transpose([2, 0, 1])
        image = image / 255
        image = torch.Tensor(image)
        image = torch.stack([image])
        res = self(image)
        res = res[0]
        res[::2] *= o_w / 512
        res[1::2] *= o_h / 512
        return res  # first 8 point (x,y) for mrz second 8 point xy for card



class MRZCheker():
    def __init__(self, text):
        self.lines = text.split("\n")
        self.lines = list(filter(lambda x: len(x)> 5, self.lines))
        self.type = "TD2"
        if len(self.lines) == 3:
            self.type = "TD1"
        elif len(self.lines[0]) > 40:
            self.type = "MRP"

    def checksum(self, segment):
        nums = [7, 3, 1]
        cur_sum = 0
        for idx,c in enumerate(segment):
            d = 0
            if  'A' <= c <= 'Z':
                d = 10 + (ord(c) - ord('A'))
            if  '0' <= c <= '9':
                d = ord(c) - ord('0')
            cur_sum += d * nums[idx%3]
        return cur_sum%10
    
        
    def check(self):
        checker = {
            "MRP": self.checkMRP,
            "TD2": self.checkTD2,
            "TD1": self.checkTD1
        }
        return checker[self.type]()
            
    
    def checkMRP(self):
        if str(self.checksum(self.lines[1][:9])) !=  MRZCheker.arrow_to_zero(self.lines[1][9]):
            return False
        if str(self.checksum(self.lines[1][13:19])) !=  MRZCheker.arrow_to_zero(self.lines[1][19]):
            return False
        if str(self.checksum(self.lines[1][21:27])) !=  MRZCheker.arrow_to_zero(self.lines[1][27]):
            return False
        if str(self.checksum(self.lines[1][28:42])) !=  MRZCheker.arrow_to_zero(self.lines[1][42]):
            return False
        return True
    
    def arrow_to_zero(c):
        if c == "<":
            return '0'
        return c
        
    def checkTD1(self):
        if str(self.checksum(self.lines[0][5:14])) != MRZCheker.arrow_to_zero(self.lines[0][14]):
            return False
        if str(self.checksum(self.lines[1][:6])) !=  MRZCheker.arrow_to_zero(self.lines[1][6]):
            return False
        if str(self.checksum(self.lines[1][8:14])) !=  MRZCheker.arrow_to_zero(self.lines[1][14]):
            return False
        return True
    
    def checkTD2(self):
        if str(self.checksum(self.lines[1][:9])) !=  MRZCheker.arrow_to_zero(self.lines[1][9]):
            return False
        if str(self.checksum(self.lines[1][13:19])) != MRZCheker.arrow_to_zero(self.lines[1][19]):
            return False
        if str(self.checksum(self.lines[1][21:27])) !=  MRZCheker.arrow_to_zero(self.lines[1][27]):
            return False
        return True


class MrzDetector():
    ARROW_MATCH_TRESH = 0.75
    MIN_ARROW_COUNT_PER_LINE = 3
    MIN_ARROW_BOX_SHAPE = (5, 5)
    MRZ_LINE_WIDTH_PERCENTAGE = 0.5
    METHOD = "DEEP"

    def __init__(self):
        if MrzDetector.METHOD == "MATCHING":
            self.arrow_template = pickle.load(open("maxinai/mrz/sample_arrow_image.pkl", "rb"))
        elif MrzDetector.METHOD == "CRAFT":
            self.craft_model = CraftModelSettings.MODEL
        elif MrzDetector.METHOD == "DEEP":
            self.model = torch.load('mse_35.pth',
                                    map_location=torch.device('cpu'))
            self.model.eval()

    def dectect_arrow_contours(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        arrow_contours = []
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            arrow_part = thresh[y:y + h, x: x + w].copy()
            arrow_part = cv2.resize(arrow_part, (64, 64))
            d1 = cv2.matchTemplate(self.arrow_template, arrow_part, cv2.TM_CCORR_NORMED)[0][0]
            if d1 < MrzDetector.ARROW_MATCH_TRESH or w < MrzDetector.MIN_ARROW_BOX_SHAPE[0] or h < \
                    MrzDetector.MIN_ARROW_BOX_SHAPE[1]:
                continue
            arrow_contours.append(contour)
        return arrow_contours

    def dectect_arrow_contours_craft(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        mask, _ = self.craft_model(image)
        _, mask = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        arrow_contours = []
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            arrow_part = thresh[y:y + h, x: x + w].copy()
            arrow_part = cv2.resize(arrow_part, (64, 64))
            d1 = cv2.matchTemplate(self.arrow_template, arrow_part, cv2.TM_CCORR_NORMED)[0][0]
            if d1 < MrzDetector.ARROW_MATCH_TRESH or w < MrzDetector.MIN_ARROW_BOX_SHAPE[0] or h < \
                    MrzDetector.MIN_ARROW_BOX_SHAPE[1]:
                continue
            arrow_contours.append(contour)
        return arrow_contours

    def get_mrz_lines(self, image, arrow_contours):
        im_h, im_w = image.shape[:2]
        arrow_boxes = list(map(cv2.boundingRect, arrow_contours))
        line_open, line_close = np.zeros(im_h), np.zeros(im_h)
        for (x, y, w, h) in arrow_boxes:
            line_open[y] += 1

            line_close[min(y + h, im_h - 1)] += 1

        cur_lines = 0
        is_new_line = True
        one_line_already_detected = False
        lines = []
        for y in range(0, im_h):
            cur_lines += line_open[y]
            if cur_lines == 0:
                is_new_line = True
            if (cur_lines >= MrzDetector.MIN_ARROW_COUNT_PER_LINE or (
                    one_line_already_detected and cur_lines > 0)) and is_new_line:
                one_line_already_detected = True
                is_new_line = False
                lines.append(y)
            cur_lines -= line_close[y]
        return lines

    def get_mrz_craft(self, image):
        arrow_contours = self.dectect_arrow_contours(image)
        line_ys = self.get_mrz_lines(image, arrow_contours)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        mask, _ = self.craft_model(image)
        _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: -cv2.boundingRect(x)[2])
        min_x, min_y = image.shape[1], image.shape[0]
        max_x, max_y = 0, 0
        for idx, contour in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour)
            if w > (1 if line_ys else 0.5) * MrzDetector.MRZ_LINE_WIDTH_PERCENTAGE * image.shape[1]:
                for line_y in line_ys:
                    if abs(y - line_y) > 4 * h:
                        continue
                    min_x = min(x, min_x)
                    min_y = min(y, min_y)
                    max_x = max(x + w, max_x)
                    max_y = max(y + h, max_y)
                    line_ys.append(y)
                    break
            else:
                for line_y in line_ys:
                    if abs(line_y - y) < h:
                        min_x = min(x, min_x)
                        min_y = min(y, min_y)
                        max_x = max(x + w, max_x)
                        max_y = max(y + h, max_y)
                        if len(line_ys) <= 1:
                            line_ys.append(y)
                        break
        return (min_x, min_y, max_x - min_x, max_y - min_y)

    def get_mrz_deep(self, image):
        o_h, o_w = image.shape[:2]
        output = self.model.predict(image)
        x1, y1 = torch.min(output[:8:2]), torch.min(output[1:8:2])
        x2, y2 = torch.max(output[:8:2]), torch.max(output[1:8:2])
        w, h = x2 - x1, y2 - y1
        x1 = max(0, x1 - 0.02 * o_w)
        y1 = max(0, y1 - 0.02 * o_h)
        x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
        mrz_part = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(mrz_part, cv2.COLOR_BGR2GRAY)
        # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 10)
        # contour_boxes = []
        # for idx, contour in enumerate(contours):
        #     (x, y, w, h) = cv2.boundingRect(contour)
        #     contour_boxes.append((x + x1, y + y1, w, h))
        crop_image = image[y1:y2, x1:x2]
        print(self.read_data(crop_image))
        return crop_image

    def read_data(self, crop_image):
        text = pytesseract.image_to_string(crop_image, lang='ocrb', config="--tessdata-dir /home/xarshila/Desktop/HS/works/Useful/tesseract/retrain")
        mrz_checker = MRZCheker(text)
        check = mrz_checker.check() 

        def mrp_extract(lines):
            result = {}
            parts = list(filter(lambda x: len(x) > 0, lines[0].split("<")))
            result["issuing_country"] = parts[1][:3] 
            result["last_name"] = parts[1][3:]
            result["first_name"] = " ".join(parts[2:])
            result["card_id"] = lines[1][:9]
            result["nation"] = lines[1][10:13]
            result["birth_date"] = lines[1][13:19]
            result["sex"] = lines[1][20]
            result["exp_date"] = lines[1][21:27]
            return result

        def td1_extract(lines):
            result = {}
            result["issuing_country"] = lines[0][2:5]
            result["card_id"] = lines[0][5:14]
            result["birth_date"] = lines[1][:6]
            result["exp_date"] = lines[1][8:14]
            result["nation"] = lines[1][15:18]
            result["sex"] = lines[1][7]
            parts = list(filter(lambda x: len(x) > 0, lines[2].split("<")))
            result["last_name"] = parts[0]
            result["first_name"] = " ".join(parts[1:])
            return result

        def td2_extract(lines):
            result = {}
            result["issuing_country"] = lines[0][2:5]
            parts = list(filter(lambda x: len(x) > 0, lines[0][5:].split("<")))
            result["last_name"] = parts[0]
            result["first_name"] = " ".join(parts[1:])
            result["card_id"] = lines[1][:9]
            result["birth_date"] = lines[1][13:19]
            result["exp_date"] = lines[1][21:27]
            result["nation"] = lines[1][10:13]
            result["sex"] = lines[1][20]
            return result

        if mrz_checker.type == "MRP":
            return mrp_extract(mrz_checker.lines), check
        elif mrz_checker.type == "TD2":
            return td2_extract(mrz_checker.lines), check
        return td1_extract(mrz_checker.lines), check

    def detect(self, image):
        if MrzDetector.METHOD == "DEEP":
            return self.get_mrz_deep(image)
        elif MrzDetector.METHOD == "CRAFT":
            return self.get_mrz_craft(image)
        else:
            arrow_contours = self.dectect_arrow_contours(image)
            mrz_lines = self.get_mrz_lines(image, arrow_contours)
        return mrz_lines

if __name__ == "__main__":
    mrzDetector = MrzDetector()
    import sys
    mrz_img = mrzDetector.detect(cv2.imread(sys.argv[1]))
    import matplotlib.pyplot as plt 
    plt.imshow(mrz_img)
    plt.show()