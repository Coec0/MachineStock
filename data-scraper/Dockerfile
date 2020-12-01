FROM ubuntu:18.04

RUN apt update
RUN apt-get install -y python3 python3-pip locales iproute2 curl iputils-ping

RUN pip3 install --upgrade pip

WORKDIR /app
COPY . .

RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN pip3 install pytz
RUN pip3 install requests

RUN chmod +x start-fetchers.sh

CMD sh start-fetchers.sh
