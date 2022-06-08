import torch
import cv2
import numpy as np
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from configparser import ConfigParser


class Model_inference_person:
    def __init__(self,
                 config_path,
                 device,
                 confidence=None
                 ):
        """
        :param config_path: путь к конфигурационному файлу
        :param device: cpu или gpu
        :param confidence: порог уверенности для обнаруженных людей
        """
        config = ConfigParser()
        config.read(config_path)
        self._device = device
        self._confidence = confidence or config.getfloat('model_person', 'confidence')
        # сделать возможность выбора сети из https://pytorch.org/vision/stable/models.html
        self._model = fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True).to(self._device).eval()

    # находит людей на фото
    # сеть ищет объекты, возвращает словарь с координатами объектов формата [x1, y1, x2, y2] - detections["boxes"],
    # confidence - detections["scores"]
    # и лейблами по coco - detections["labels"](человек - 1)
    # ДОРАБОТАТЬ при случаях, когда людей на фото много
    def person_in_the_photo(self, image=None, return_detections=False):
        is_person = False
        image = self._image_conversion(image)
        detections = self._model(image)[0]
        for i in range(0, len(detections["boxes"])):
            confidence = detections["scores"][i]
            idx = int(detections["labels"][i])
            if idx == 1 and confidence >= self._confidence:
                is_person = True
                break
        if return_detections:
            return is_person, detections
        else:
            return is_person

    # рисует прямоугольники для людей
    def draw_rectangles(self, image, detections):
        green_color = [0, 255, 0]
        for i in range(0, len(detections["boxes"])):
            label = int(detections["labels"][i])
            if label == 1:
                confidence = detections["scores"][i]
                if confidence > self._confidence:
                    # tensor([ 21.7621, 91.3633, 312.5862, 492.4698], grad_fn=<SelectBackward0>) ->
                    # -> [ 21.7621, 91.3633, 312.5862, 492.4698]
                    box = detections["boxes"][i].detach().numpy()
                    startX = int(box[0])
                    startY = int(box[1])
                    endX = int(box[2])
                    endY = int(box[3])
                    cv2.rectangle(image, (startX, startY), (endX, endY), green_color, 2)
        return image

    # конвертирует изображение для сети
    # взято с https://pyimagesearch.com/2021/08/02/pytorch-object-detection-with-pre-trained-networks/
    def _image_conversion(self, image):
        image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image_RGB.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        image = torch.FloatTensor(image)
        image = image.to(self._device)
        return image