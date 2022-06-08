import logging
import json
from src.video_processing import Video_processing


if __name__ == '__main__':
    # конфигурация логирования, уровень записи info
    logging.basicConfig(level=logging.INFO,
                        filename='files_output/log.txt',
                        filemode='w',
                        format='%(asctime)s %(message)s')
    video_processing = Video_processing(config_path='files/config.ini')
    time_output, frame_output = video_processing.video_processing(video_path='files/2.mp4',
                                                    image_path='files/2.jpg')
    with open('files_output/time_output.json', 'w') as js:
        text = json.dumps({"time_output": time_output, "frame_output": frame_output})
        js.write(text)







