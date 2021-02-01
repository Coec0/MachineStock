FROM ubuntu:18.04

RUN apt update
RUN apt-get install -y python3 python3-pip

RUN pip install --upgrade pip
RUN pip install torch torchvision numpy

WORKDIR /app
COPY python-docker/ .

CMD python3 test.py
