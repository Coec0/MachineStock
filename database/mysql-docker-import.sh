#!/bin/bash
. config.sh

docker exec -i $DOCKER_NAME sh -c 'exec mysql -uroot -p'$ROOT_PASSWORD < $BACKUP_FILE
