#include <cstdlib>
#include <iostream>

#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include <cppconn/resultset.h>
#include <cppconn/statement.h>
#include <mysql_driver.h>
#include <sstream>
#include <string>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <chrono>
#include <thread>
#include "network_connection.h"

using namespace std::chrono;
using namespace std;
using namespace rapidjson;

string sql_row_to_json(const char *stock, const char *sector, int publication_time, const char *mmt_flags,
                       const char *transaction_id_code, double price, int volume);

void parse_arguments(int argc, char *argv[]);

int port = 2000;
int mysql_port = 3306;
string mysql_ip;
int epoch_start = 1606989600;//Thursday, December 3, 2020 10:00:00
int epoch_end = 1607007600;
double time_factor = 1;
string stocks;
string sectors;
string username;
string password;

void error_parse(const char *msg) {
    perror(msg);
    exit(EXIT_FAILURE);
}

string comma_to_sql(const string &values);

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
int main(int argc, char *argv[]) {
    mysql_ip = "127.0.0.1";
    parse_arguments(argc, argv);
    int simulated_time = epoch_start;
    network_connection connection(port);

    try {
        sql::mysql::MySQL_Driver *driver;
        sql::Connection *con;
        sql::Statement *stmt;
        sql::ResultSet *res;

        cout << "Connecting to sql server " << endl;
        driver = sql::mysql::get_driver_instance();
        stringstream connection_ss;
        connection_ss << "tcp://" + mysql_ip + ":" << mysql_port;

        con = driver->connect(connection_ss.str(), username, password); //mosigtedson root
        con->setSchema("orders");

        cout << "Connected! " << endl;
        stmt = con->createStatement();
        std::stringstream ss;
        ss << "SELECT * FROM market_orders "
              "WHERE (publication_time BETWEEN " << epoch_start << " AND " << epoch_end << ") ";
        if (!stocks.empty())
            ss << "AND stock in (" << comma_to_sql(stocks) << ") ";
        if (!sectors.empty() && stocks.empty())
            ss << "AND sector in (" << comma_to_sql(sectors) << ") ";

        ss << "ORDER BY publication_time ASC , transaction_id_code ASC ";

        cout << "Querying data using \"" + ss.str() + "\".\nThis can take some time..." << endl;
        res = stmt->executeQuery(ss.str());
        cout << "Data fetched! Rows matched: " << res->rowsCount() << endl;
        cout << "Waiting for client..." << endl;
        connection.listen_socket();
        cout << "Client connected! Starting transmission" << endl;
        auto start = high_resolution_clock::now(); //Makes sure its initialised
        while (res->next()) {
            while (simulated_time < res->getInt("publication_time")) { //Wait until correct timestamp
                auto stop = high_resolution_clock::now(); //Stop timer
                long duration = duration_cast<microseconds>(stop - start).count(); //Calculate delta from start of timer
                int sleep_time = 1000000 * time_factor;
                if (duration > sleep_time)
                    cout << simulated_time << ": Cannot' keep up! Delayed by "
                         << duration - sleep_time
                         << " microseconds"
                         << endl;
                std::this_thread::sleep_for(std::chrono::microseconds(
                        max(0l, sleep_time - duration))); // Remove delta from 1 second to get correct seconds
                simulated_time++;
                start = high_resolution_clock::now(); //Start timer to see how long execution time takes
            }
            const string json = sql_row_to_json(
                    res->getString("stock").c_str(),
                    res->getString("sector").c_str(),
                    res->getInt("publication_time"),
                    res->getString("mmt_flags").c_str(),
                    res->getString("transaction_id_code").c_str(),
                    res->getDouble("price"),
                    res->getInt("volume")
            );
            connection.write_to_socket(json.c_str());
        }
        delete res;
        delete stmt;
        delete con;

    } catch (sql::SQLException &e) {
        cout << "# ERR: SQLException in " << __FILE__;
        cout << "(" << __FUNCTION__ << ") on line " << __LINE__ << endl;
        cout << "# ERR: " << e.what();
        cout << " (MySQL error code: " << e.getErrorCode();
        cout << ", SQLState: " << e.getSQLState() << " )" << endl;
    }

    cout << endl;

    return EXIT_SUCCESS;
}

string sql_row_to_json(const char *stock, const char *sector, int publication_time, const char *mmt_flags,
                       const char *transaction_id_code, double price, int volume) {
    StringBuffer s;
    Writer<StringBuffer> writer(s);
    writer.StartObject();               // Between StartObject()/EndObject(),
    writer.Key("stock");                // output a key,
    writer.String(stock);             // follow by a value.
    writer.Key("sector");
    writer.String(sector);
    writer.Key("publication_time");
    writer.Int(publication_time);
    writer.Key("mmt_flags");
    writer.String(mmt_flags);
    writer.Key("transaction_id_code");
    writer.String(transaction_id_code);
    writer.Key("price");
    writer.Double(price);
    writer.Key("volume");
    writer.Int(volume);
    writer.Key("timestamp_ms");
    auto now = high_resolution_clock::now();
    long timestamp = duration_cast<milliseconds>(now.time_since_epoch()).count();
    writer.Int64(timestamp);
    writer.EndObject();

    const char *ret_val = s.GetString();
    string ret_string(ret_val);
    return ret_string;
}

void parse_arguments(int argc, char *argv[]) {
    int real_args = argc - 1; //Remove program name
    if (real_args % 2 != 0)
        error_parse("Uneven number of arguments. Exiting...");
    for (int i = 1; i < argc; i = i + 2) {
        char *arg = argv[i];
        if (strcmp(arg, "--mysql-user") == 0)
            username = argv[i + 1];
        else if (strcmp(arg, "--mysql-pass") == 0)
            password = argv[i + 1];
        else if (strcmp(arg, "--port") == 0)
            port = stoi(argv[i + 1]);
        else if (strcmp(arg, "--mysql-ip") == 0)
            mysql_ip = argv[i + 1];
        else if (strcmp(arg, "--mysql-port") == 0)
            mysql_port = stoi(argv[i + 1]);
        else if (strcmp(arg, "--stocks") == 0) {
            if (strcmp(argv[i + 1], "\"\"") != 0 && strcmp(argv[i + 1], "''") != 0)
                stocks = argv[i + 1];
        } else if (strcmp(arg, "--sectors") == 0) {
            if (strcmp(argv[i + 1], "\"\"") != 0 && strcmp(argv[i + 1], "''") != 0)
                sectors = argv[i + 1];
        } else if (strcmp(arg, "--epoch-start") == 0)
            epoch_start = stoi(argv[i + 1]);
        else if (strcmp(arg, "--epoch-end") == 0)
            epoch_end = stoi(argv[i + 1]);
        else if (strcmp(arg, "--time-factor") == 0)
            time_factor = stod(argv[i + 1]);
        else
            error_parse("Unknown argument supplied. Exiting...");
    }
    if (username.empty() || password.empty()) {
        error_parse(R"(Missing "--mysql-user" or "--mysql-pass". Exiting...)");
    }
}

string comma_to_sql(const string &values) {
    std::stringstream ss;
    istringstream ss_split(values);
    string token;
    while (getline(ss_split, token, ',')) {
        ss << "\"" << token << "\",";
    }
    string combined = ss.str();
    combined.pop_back();
    return combined;
}
