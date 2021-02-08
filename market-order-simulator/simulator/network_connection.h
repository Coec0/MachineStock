
#ifndef SIMULATOR_NETWORK_CONNECTION_H
#define SIMULATOR_NETWORK_CONNECTION_H

#include <string>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>

using namespace std;

class network_connection {
public:
    explicit network_connection(int port);
    virtual ~network_connection();

    void listen_socket();
    void write_to_socket(const char *message);

private:
    int socket_fd, new_socket_fd;
};


#endif //SIMULATOR_NETWORK_CONNECTION_H
