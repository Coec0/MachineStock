FROM ubuntu:20.04

RUN apt-get update
RUN apt-get install -y python3 python3-pip

RUN pip3 install --upgrade pip
RUN pip install torch numpy mysql-connector-python

WORKDIR /app
COPY python-docker/ .

CMD python3 main.py
