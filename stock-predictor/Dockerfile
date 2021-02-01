FROM ubuntu:18.04

RUN apt update
RUN apt-get install -y python3 python3-pip

RUN pip3 install --upgrade pip
RUN pip install torch torchvision numpy mysql-connector-python

WORKDIR /app
COPY python-docker/ .

CMD python3 test.py
