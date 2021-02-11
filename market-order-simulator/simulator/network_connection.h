
#ifndef SIMULATOR_NETWORK_CONNECTION_H
#define SIMULATOR_NETWORK_CONNECTION_H

#include <string>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <forward_list>
#include <tuple>
#include <set>
#include <rapidjson/writer.h>

using namespace std;

class network_connection {
public:
    explicit network_connection(int port);
    virtual ~network_connection();

    void listen_socket();
    void write_to_socket(const char * message);

private:
    [[noreturn]] void thread_socket();
    int socket_fd;
    forward_list<tuple<int, set<string>>> sockets;
    thread t;
};


#endif //SIMULATOR_NETWORK_CONNECTION_H
