## Install using docker

### 1. Copy `env-example.list` to `env.list`

The `env.list` contains all arguments that the simulator will use. `mysql_user=` and `mysql_pass=` must be set! The possible arguments are the following:

``` 
mysql_user=<mysql username>
mysql_pass=<mysql password>
port=<listen port (default 2000)>
mysql_ip=<ip to mysql server (default 127.0.0.1)>
mysql_port=<port to mysql server (default 3306)>
stocks=<stocks separated with ",". Leave empty for all stocks. This overrides sectors if both are set. Example: Handelsbanken,Avanza>
sectors=<sectors separated with ",". Leave empty for all sectors. Example: financials,healthcare>
epoch_start=<the starting simulated time measured in seconds since 1970 (default 1606989600)>
epoch_end=<the ending in simulated time measured in seconds since 1970 (default 1607007600)>
time_factor=<sleep time is modified with this factor. 0.25 would result in 4 times speedup>
```

### 2. Build docker file

`docker build -t stock-simulator:latest .`

### 3. Run docker

`docker run --network host --env-file ./env.list --name stock_simulator -it stock-simulator`

## Run standalone

### 1. Install required packages

`apt-get install -y libmysqlcppconn-dev cmake g++`

### 2. Make file

```
cmake .
make
```

### 3. Run simulator

Run the simulator with selected arguments, such as:

`./simulator --mysql-user root --mysql-pass toor`

Available arguments are:

```
/**
 * Arguments:
 *
 * Required:
 *
 * --mysql-user <mysql username>
 * --mysql-pass <mysql password>
 *
 * Optional:
 *
 * --port <listen port (default 2000)>
 * --mysql-ip <ip to mysql server (default 127.0.0.1)>
 * --mysql-port <port to mysql server (default 3306)>
 * --stocks <stocks separated with ",". Leave empty for all stocks. This overrides sectors if both are set. Example:
 *      "Handelsbanken,Avanza">
 * --sectors <sectors separated with ",". Leave empty for all sectors. Example: "financials,healthcare">
 * --epoch-start <the starting simulated time measured in seconds since 1970 (default 1606989600)>
 * --epoch-end <the ending in simulated time measured in seconds since 1970 (default 1607007600)>
 * --time-factor <sleep time is modified with this factor. 0.25 would result in 4 times speedup>
 */
```
