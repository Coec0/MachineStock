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

using namespace std;
using namespace rapidjson;

string sql_row_to_json(const char* stock, const char* sector, int publication_time, const char* mmt_flags,
                             const char* transaction_id_code, double price, int volume);

int main()
{
    int epoch_start = 1606867200; //Wednesday, December 2, 2020 0:00:00
    int epoch_end = 1606953600;
    try {
        sql::mysql::MySQL_Driver *driver;
        sql::Connection *con;
        sql::Statement *stmt;
        sql::ResultSet *res;

        driver = sql::mysql::get_driver_instance();
        con = driver->connect("tcp://127.0.0.1:3306", "root", "mosigtedson");
        con->setSchema("orders");

        stmt = con->createStatement();
        std::stringstream ss;
        ss << "SELECT * FROM market_orders "
             "WHERE publication_time BETWEEN " << epoch_start << " AND " << epoch_end <<
             " ORDER BY publication_time ASC LIMIT 0,10";
        res = stmt->executeQuery(ss.str()); // replace with your statement
        cout << "Size " << res->rowsCount() << endl;
        while (res->next()) {
            const string json = sql_row_to_json(
                    res->getString("stock").c_str(),
                    res->getString("sector").c_str(),
                    res->getInt("publication_time"),
                    res->getString("mmt_flags").c_str(),
                    res->getString("transaction_id_code").c_str(),
                    res->getDouble("price"),
                    res->getInt("volume")
                    );
            cout << json << endl;
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

string sql_row_to_json(const char* stock, const char* sector, int publication_time, const char* mmt_flags,
                             const char* transaction_id_code, double price, int volume){
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

    const char* ret_val = s.GetString();
    string ret_string (ret_val);
    return ret_string;
}
