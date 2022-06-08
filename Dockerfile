FROM python:3.8
RUN pip3 install --upgrade pip
COPY files/requirements.txt /root/requirements.txt
RUN python3 -m pip install -r /root/requirements.txt
RUN apt-get update && apt-get install libgl1 -y
#RUN apt-get install ffmpeg libsm6 libxext6  -y
WORKDIR /root/face_recognitition
COPY . /root/face_recognitition
RUN cd /root/face_recognitition
RUN mkdir files_output
CMD ./start.sh