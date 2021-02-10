cmake_minimum_required(VERSION 3.16)
project(simulator)

set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

add_executable(simulator main.cpp network_connection.cpp network_connection.h)
target_link_libraries(simulator mysqlcppconn)
include_directories(include)