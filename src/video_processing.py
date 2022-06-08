from src.model_inferance_face import Model_inference_face
from src.model_inferance_person import Model_inference_person
import cv2
import logging
import datetime
from configparser import ConfigParser
import torch
import json

class Video_processing:
    def __init__(self,
                 config_path,
                 skipped_seconds=None,
                 smoothing_threshold=None
                 ):
        """

        :param config_path: конфигурационный файл
        :param skipped_seconds: сколько секунд пропускаем между проверками, если 0, то будет обробатываться каждый фрейм
        :param smoothing_threshold: сколько максимально может быть skipped_seconds между отрезками времени
        например, если skipped_seconds = 2, smoothing_threshold = 3,
        то между отрезками времени должно быть не бльше 6 секунд, тогда их можно соединить
        smothing_threshold не может быть нулем!
        """
        config = ConfigParser()
        config.read(config_path)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._smoothing_threshold = smoothing_threshold or config.getint('video', 'smoothing_threshold')
        self._skipped_seconds = skipped_seconds or config.getint('video', 'skipped_seconds')
        self._model_face = Model_inference_face(config_path=config_path, device=device)
        self._model_person = Model_inference_person(config_path=config_path, device=device)
        self._output = []

    def video_processing(self, video_path, image_path, save_path_face1=None):
        self._output = []
        # определяем вектор лица с картинки
        face1_vector = self._model_face.detect_face(image_path=image_path, save_path=save_path_face1)
        if face1_vector is None:
            logging.error(f'Лицо с фото {image_path} не прочитано')
            assert False, f"Лицо с фото {image_path} не прочитано"


        # начинаем считывать видео
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        # считаем сколько кадров надо пропустить
        if self._skipped_seconds != 0:
            frame_skipping = round(fps * self._skipped_seconds)
        else:
            frame_skipping = 1
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_id = int(cap.get(1)) - 1  # CAP_PROP_POS_FRAMES
                millisec = cap.get(0)  # CAP_PROP_POS_MSEC
                if frame_id % frame_skipping == 0:
                    face2_vectors, prob = self._model_face.detect_face(image=frame,
                                                                       # save_path=f'files/video_scrin/{frame_id}.jpg',
                                                                       return_prob=True)
                    if face2_vectors is None:
                        # проверяем есть ли человек, если есть, то добавляем в output
                        self._check_person(frame, frame_id, millisec, frame_skipping)
                    else:
                        similar = False
                        # проходим по всем найденным лицам
                        for i in range(len(face2_vectors)):
                            # сравниваем каждое лицо, True если меньше similarity_threshold
                            similar, distance = self._model_face.compare_faces(face2_vectors[i], face1_vector[0])
                            if similar:
                                self._add_frame_to_output(frame_id, millisec, frame_skipping)
                                logging.info(f'Frame_id:{frame_id}, mlsec:{millisec}. '
                                             f'Лицо: +, совпадение: + . prob:{prob[i]}')
                                # cv2.imwrite(f'video_scrin/{frame_id}_меньше_порога.jpg', frame)
                                break
                        if not similar:
                            logging.info(f'Frame_id:{frame_id}, mlsec:{millisec}. '
                                         f'Лицо: +, совпадение: - . distance:{distance}')
                            # cv2.imwrite(f'video_scrin/{frame_id}больше1.jpg', frame)
            # добавить обработку нечитающихся фреймов
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
        time_output, frame_output = self._output_to_time(self._output)
        print(time_output)
        logging.info(frame_output)
        logging.info(time_output)
        return time_output, frame_output

    # проверяет есть ли человек на фото, если есть, то заносит в output
    def _check_person(self, frame, frame_id, millisec, frame_skipping):
        is_person, detections = self._model_person.person_in_the_photo(image=frame,
                                                                       return_detections=True)
        # для отладки.
        # img = self._model_person.draw_rectangles(frame, detections, save_path = )
        # cv2.imwrite(f'video_scrin/{frame_id}.jpg', img)
        if is_person:
            self._add_frame_to_output(frame_id, millisec, frame_skipping)
            logging.info(f'Frame_id:{frame_id}, mlsec:{millisec}. '
                         f'Лицо: - , челвоек: +')
        else:
            logging.info(f'Frame_id:{frame_id}, mlsec:{millisec}. '
                         f'Лицо: -, челвоек: -')

    # добавляет фрейм в output. если новый фрейм отличается от последнего в списке на frame_skipping,
    # то последний заменяется прибывшим, иначе начинается новый отрезок
    # output имеет вид [[frame_id_start, frame_id_end, millisec_start, millisec_end],
    #                    [.., .., .., ..,]]
    def _add_frame_to_output(self, frame, millisec, frame_skipping):
        if self._output == []:
            self._output.append([frame, frame, millisec, millisec])
        else:
            last_frame_id = len(self._output) - 1
            last_frame = self._output[last_frame_id][1]

            if frame - last_frame <= frame_skipping*self._smoothing_threshold:
                self._output[last_frame_id][1] = frame
                self._output[last_frame_id][3] = millisec
            else:
                self._output.append([frame, frame, millisec, millisec])

    def _output_to_time(self, output):
        time_output = []
        frame_output = []
        for i in range(len(output)):
            start_millisec = output[i][2]
            start_time = str(datetime.timedelta(milliseconds=start_millisec))
            end_millisec = output[i][3]
            end_time = str(datetime.timedelta(milliseconds=end_millisec))

            start_frame = output[i][0]
            end_frame = output[i][1]
            time_output.append([start_time, end_time])
            frame_output.append([start_frame, end_frame])
        return time_output, frame_output

    # делаем новое видео
    def cropped_video(self, video_path, json_path, save_path):
        cropped_video = None
        with open(json_path, 'r') as f:
            file = json.load(f)
        # выбираем фреймы
        frames = file["frame_output"]
        # начинаем читать видео
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        # id текущего отрезка
        frames_id = 0
        # id последнего отрезка
        frames_id_last = len(frames)-1
        # текущий номер фрейма в текущем отрезке
        frame_id_cur = frames[frames_id][0]
        # последний номер фрейма в текущем отрезке
        last_frame_id_cur = frames[frames_id][1]
        while cap.isOpened():
            ret, frame = cap.read()
            # создаем видео
            if cropped_video is None:
                cropped_video = self._create_video(frame, fps, save_path)
            if ret:
                frame_id = int(cap.get(1)) - 1  # текущий фрейм обробатываемого видео
                # добавляем кадр, если текущий фрейм обробатываемого видео совпадает с текущим фреймом в текущем отрезке
                if frame_id == frame_id_cur:
                    cropped_video.write(frame)
                    # если это не последний фрейм в текущем отрезке, то переходим к следующему
                    if frame_id_cur != last_frame_id_cur:
                        frame_id_cur = frame_id_cur + 1
                    # если это последний фрейм в текущем отрезке, то
                    else:
                        # если отрезки закончились, выходим
                        if frames_id == frames_id_last:
                            break
                        # если не закончились, то переходим к новому фрейму
                        else:
                            frames_id = frames_id+1
                            frame_id_cur = frames[frames_id][0]
                            last_frame_id_cur = frames[frames_id][1]

    # инициализируем видео
    def _create_video(self, frame, fps, save_path):
        height = frame.shape[0]
        width = frame.shape[1]
        cropped_video = cv2.VideoWriter(save_path,
                                        cv2.VideoWriter_fourcc(*'XVID'),
                                        fps,
                                        (width, height))
        return cropped_video


