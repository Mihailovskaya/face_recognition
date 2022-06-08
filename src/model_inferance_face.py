from facenet_pytorch import MTCNN
import torch
import cv2
from facenet_pytorch import InceptionResnetV1
from configparser import ConfigParser
import logging

class Model_inference_face:

    def __init__(self,
                 config_path,
                 device,
                 pretrained=None,
                 face_confidence=None,
                 similarity_threshold=None
                 ):
        """
        :param config_path: путь к конфигурационному файлу
        :param device: cpu или gpu
        :param pretrained: датасет на котором обучалась InceptionResnetV1. VGGFace2 или CASIA-Webface
        :param face_confidence: порог confidence для вывода лица из модели
        :param similarity_threshold: порог похожести: обычно меньше 1 - лица схожи
        """
        config = ConfigParser()
        config.read(config_path)

        self._face_confidence = face_confidence or config.getfloat('model_face', 'face_confidence')
        self._device = device
        # модель ищет лица на изображении
        # возвращает вектора обнаруженных лиц и вероятность, если нужно
        # больше: help(MTCNN)
        self._mtcnn = MTCNN(
            keep_all=True,  # вернуть все лица на изображении
            select_largest=False,  # Если False, возвращается лицо с наибольшей вероятностью обнаружения.
            thresholds=[self._face_confidence, self._face_confidence, self._face_confidence],
            device=self._device
        )

        # датасет для предобученной модели InceptionResnetV1
        # VGGFace2 или CASIA-Webface
        self.pretrained = pretrained or config.get('model_face', 'pretrained')
        self._similarity_threshold = similarity_threshold or config.getfloat('model_face', 'similarity_threshold')
        # модель считает image embeddings, после этого считаем растояние между двумя embeddings и получаем "схожесть"
        # алгоритм взят, узнать больше: https://github.com/timesler/facenet-pytorch/blob/master/examples/infer.ipynb
        # так же help(InceptionResnetV1)
        self._resnet = InceptionResnetV1(pretrained=self.pretrained, device=self._device).eval()

    # обноруживает лицо, может сохранить обрезанное, если указать путь
    # возвращает вектора найденных лиц и их вероятность, если нужно
    def detect_face(self, image_path=None, image=None, save_path=None, return_prob=False):
        if image is None:
            image = cv2.imread(image_path)
        if image is None:
            logging.error(f'фото {image_path} не прочитано')
            assert False, f"фото {image_path} не прочитано"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if return_prob:
            face_vectors, prob = self._mtcnn(image, save_path=save_path, return_prob=True)
            return face_vectors, prob
        else:
            face_vectors = self._mtcnn(image, save_path=save_path)
            return face_vectors

    # сравнивает два лица, принимает вектора лиц
    # возвращает bool одинаковые ли лица и растояние
    # считает embeddings и дистанцию между ними
    # решает по границе - similarity_threshold, граница выбрана опытным путем :) возможно требует доработки
    def compare_faces(self, face1, face2):
        similar = False
        face_vectors = [face1, face2]
        aligned = torch.stack(face_vectors).to(self._device)
        embeddings = self._resnet(aligned).detach().to(self._device)
        distance = (embeddings[0] - embeddings[1]).norm().item()
        if distance < self._similarity_threshold:
            similar = True
        return similar, distance
