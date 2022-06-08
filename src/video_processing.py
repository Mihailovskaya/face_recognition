from src.model_inferance_face import Model_inference_face
from src.model_inferance_person import Model_inference_person
import cv2
import logging
import datetime
from configparser import ConfigParser
import torch

class Video_processing:
    def __init__(self,
                 config_path,
                 skipped_seconds=None
                 ):
        config = ConfigParser()
        config.read(config_path)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
                                logging.info(f'Frame_id:{frame_id}, mlsec{millisec}. '
                                             f'Лицо: +, совпадение: + . prob:{prob[i]}')
                                # cv2.imwrite(f'video_scrin/{frame_id}_меньше_порога.jpg', frame)
                                break
                        if not similar:
                            logging.info(f'Frame_id:{frame_id}, mlsec{millisec}. '
                                         f'Лицо: +, совпадение: - . distance:{distance}')
                            # cv2.imwrite(f'video_scrin/{frame_id}больше1.jpg', frame)
            # добавить обработку нечитающихся фреймов
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
        logging.info(self._output)
        time_output = self._output_to_time(self._output)
        print(time_output)
        logging.info(time_output)
        return time_output

    # проверяет есть ли человек на фото, если есть, то заносит в output
    def _check_person(self, frame, frame_id, millisec, frame_skipping):
        is_person, detections = self._model_person.person_in_the_photo(image=frame,
                                                                       return_detections=True)
        # для отладки.
        # img = self._model_person.draw_rectangles(frame, detections, save_path = )
        # cv2.imwrite(f'video_scrin/{frame_id}.jpg', img)
        if is_person:
            self._add_frame_to_output(frame_id, millisec, frame_skipping)
            logging.info(f'Frame_id:{frame_id}, mlsec{millisec}. '
                         f'Лицо: - , челвоек: +')
        else:
            logging.info(f'Millisec:{frame_id}, mlsec{millisec}. '
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

            if frame - last_frame == frame_skipping:
                self._output[last_frame_id][1] = frame
                self._output[last_frame_id][3] = millisec
            else:
                self._output.append([frame, frame, millisec, millisec])

    def _output_to_time(self, output):
        time_output = []
        for i in range(len(output)):
            start_millisec = output[i][2]
            start_time = str(datetime.timedelta(milliseconds=start_millisec))
            end_millisec = output[i][3]
            end_time = str(datetime.timedelta(milliseconds=end_millisec))
            time_output.append([start_time, end_time])
        return time_output


