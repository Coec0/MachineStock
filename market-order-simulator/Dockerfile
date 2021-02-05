FROM ubuntu:20.04
ENV mysql_user=
ENV mysql_pass=
ENV port=2000
ENV mysql_ip=127.0.0.1
ENV mysql_port=3306
ENV stocks=""
ENV sectors=""
ENV epoch_start=1606989600
ENV epoch_end=1607007600
ENV time_factor=1

RUN apt update
RUN apt-get install -y libmysqlcppconn-dev cmake g++

WORKDIR /simulator
COPY simulator/ /simulator/

RUN ls -l
RUN cmake -D CMAKE_BUILD_TYPE=Release .
RUN make

RUN chmod +x simulator

CMD ./simulator --mysql-user $mysql_user --mysql-pass $mysql_pass --port $port --mysql-ip $mysql_ip --mysql-port $mysql_port --stocks $stocks --sectors $sectors --epoch-start $epoch_start --epoch-end $epoch_end --time-factor $time_factor
