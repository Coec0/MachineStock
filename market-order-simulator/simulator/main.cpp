#include <cstdlib>
#include <iostream>
#include <mysql_connection.h>

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

int main() {
    const int epoch_start = 1606989600; //Thursday, December 3, 2020 10:00:00
    const int epoch_end = 1607007600;
    int simulated_time = epoch_start;
    int port = 2002;

    network_connection connection(port);

    try {
        sql::mysql::MySQL_Driver *driver;
        sql::Connection *con;
        sql::Statement *stmt;
        sql::ResultSet *res;

        cout << "Connecting to sql server " << endl;
        driver = sql::mysql::get_driver_instance();
        con = driver->connect("tcp://127.0.0.1:3306", "root", "mosigtedson");
        con->setSchema("orders");

        cout << "Connected! " << endl << "Querying data. This can take some time..." << endl;
        stmt = con->createStatement();
        std::stringstream ss;
        ss << "SELECT * FROM market_orders "
              "WHERE publication_time BETWEEN " << epoch_start << " AND " << epoch_end <<
           " ORDER BY publication_time ASC LIMIT 0,5000";
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
                if (duration > 1000000)
                    cout << simulated_time << ": Cannot' keep up! Delayed by "
                         << duration - 1000000
                         << " microseconds"
                         << endl;
                std::this_thread::sleep_for(std::chrono::microseconds(
                        max(0l, 1000000 - duration))); // Remove delta from 1 second to get correct seconds
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
    writer.EndObject();

    const char *ret_val = s.GetString();
    string ret_string(ret_val);
    return ret_string;
}
