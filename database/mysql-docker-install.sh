#!/bin/bash
. config.sh

docker run --name $DOCKER_NAME -p $PORT:3306 -e MYSQL_ROOT_PASSWORD=$ROOT_PASSWORD -d mysql:latest 
