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


using namespace std;


int main()
{
    int epoch_start = 1606867200; //Wednesday, December 2, 2020 0:00:00
    int epoch_end = 1606953600;
    try {
        sql::mysql::MySQL_Driver *driver;
        sql::Connection *con;
        sql::Statement *stmt;
        sql::ResultSet *res;

        /* Create a connection */
        driver = sql::mysql::get_driver_instance();
        con = driver->connect("tcp://127.0.0.1:3306", "root", "mosigtedson");
        /* Connect to the MySQL test database */
        con->setSchema("orders");

        stmt = con->createStatement();
        std::stringstream ss;
        ss << "SELECT * FROM market_orders "
             "WHERE publication_time BETWEEN " << epoch_start << " AND " << epoch_end <<
             " ORDER BY publication_time ASC LIMIT 0,10";
        res = stmt->executeQuery(ss.str()); // replace with your statement
        cout << "Size " << res->rowsCount();
        while (res->next()) {
            cout << "\t... MySQL replies: ";
            /* Access column data by alias or column name */
            cout << res->getString("publication_time") << endl;
            cout << "\t... MySQL says it again: ";
            /* Access column fata by numeric offset, 1 is the first column */
            cout << res->getString(1) << endl;
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
