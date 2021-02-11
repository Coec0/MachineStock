
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include "network_connection.h"
#include <tuple>
#include <sstream>
#include <rapidjson/writer.h>
#include <rapidjson/document.h>

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
    for (tuple<int, set<basic_string<char>, less<basic_string<char>>, allocator<basic_string<char>>>> socket : sockets) {
        close(get<0>(socket));
    }
    close(socket_fd); //This should also kill the thread as it will throw an exception
}

void network_connection::listen_socket() {
    t = std::thread(&network_connection::thread_socket, this);
}

void network_connection::write_to_socket(const char * message) {
    rapidjson::Document d;
    d.Parse(message);

    auto before = sockets.before_begin();
    uint32_t len = htonl(strlen(message));
    for (auto socket_it = sockets.begin(); socket_it != sockets.end(); ) {
        set<string> stocks = get<1>(*socket_it);
        rapidjson::Value& s = d["stock"];
        if(stocks.find(s.GetString()) != stocks.end() || stocks.empty()){
            //do whatever

            int n = write(get<0>(*socket_it), &len, sizeof(len));
            int o = write(get<0>(*socket_it), message, strlen(message));
            if (n < 0 || o < 0){
                cout << strerror(errno) << ". Most likely a client disconnected. Continuing operation..." << endl;
                close(get<0>(*socket_it));
                socket_it = sockets.erase_after(before);
            } else {
                before = socket_it;
                ++socket_it;
            }
        } else {
            before = socket_it;
            ++socket_it;
        }
    }
}

[[noreturn]] void network_connection::thread_socket() {
    try {
        while(true) {
            struct sockaddr_in cli_addr;
            listen(socket_fd, 5);
            socklen_t cli_len = sizeof(cli_addr);
            int new_socket_fd = accept(socket_fd,
                                       (struct sockaddr *) &cli_addr,
                                       &cli_len);
            if (new_socket_fd < 0)
                error("ERROR on accept");
            int msg_length_raw;
            int result = read(new_socket_fd, &msg_length_raw, 4);
            if(result < 0)
                error("ERROR on read msg length");
            int msg_length = ntohl(msg_length_raw);
            char buffer[msg_length];
            int stocks_result = read(new_socket_fd, buffer, msg_length);
            string rcv;
            rcv.append(buffer);
            set<string> stocks_list;
            istringstream f(rcv);
            string s;
            while (getline(f, s, ';')) {
                stocks_list.insert(s);
            }
            sockets.push_front(make_tuple(new_socket_fd, stocks_list));
            if(stocks_result < 0)
                error("ERROR on reading stocks");

            std::cout << "Client accepted!"<< endl;

        }
    } catch (...) {
        cout << "Error in listener. Killing thread..." << endl;
        std::terminate(); //Kill thread on exception
    }

}
