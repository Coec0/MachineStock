
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "network_connection.h"

using namespace std;
void error(const char *msg)
{
    perror(msg);
    exit(1);
}

network_connection::network_connection(int port) {
    struct sockaddr_in serv_addr;
    socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd < 0)
        error("ERROR opening socket");
    bzero((char *) &serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(port);
    int option = 1;
    setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR, &option, sizeof(option));
    if (bind(socket_fd, (struct sockaddr *) &serv_addr,
             sizeof(serv_addr)) < 0)
        error("ERROR on binding");
}
network_connection::~network_connection() {
    close(new_socket_fd);
    close(socket_fd);
}

void network_connection::listen_socket() {
    struct sockaddr_in cli_addr;
    listen(socket_fd,5);
    socklen_t cli_len = sizeof(cli_addr);
    new_socket_fd = accept(socket_fd,
                           (struct sockaddr *) &cli_addr,
                           &cli_len);
    if (new_socket_fd < 0)
        error("ERROR on accept");
}

void network_connection::write_to_socket(const char *message) {
    uint32_t len = htonl(strlen(message));
    int n = write(new_socket_fd, &len, sizeof(len));
    int o = write(new_socket_fd, message, strlen(message));
    if (n < 0 || o < 0) error("ERROR writing to socket");
}
