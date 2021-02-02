# Install Process

## 1. Config

Config is available in `config.sh`. Current parameters are:

```bash
readonly PORT=3306 #The port to be used
readonly ROOT_PASSWORD=mosigtedson #Root password. Make sure to change if database is not local
readonly DOCKER_NAME=mysql-orders #The name of the docker container
readonly BACKUP_FILE=orders.sql #The name of the backup file to be imported
```

## 2. Install MySql

Run chmod if necessary to be able to execute mysql-docker-install.sh

```bash
chmod +x mysql-docker-install.sh
sudo ./mysql-docker-install.sh
```

## 3. Import Database

Run chmod if necessary to be able to execute mysql-docker-import.sh

```bash
chmod +x mysql-docker-import.sh
sudo ./mysql-docker-import.sh
```
This might take a while
