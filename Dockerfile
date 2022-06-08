FROM python:3.8

COPY files/requirements.txt /root/requirements.txt
RUN pip3 install --upgrade pip
RUN python3 -m pip install -r /root/requirements.txt

COPY . /root/face_recognition
RUN cd /root/face_recognition
RUN python3 main.py
