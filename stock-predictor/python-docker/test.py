print("hejhejhej")
import torch
x = torch.rand(5, 3)
print(x)

import mysql.connector

cnx = mysql.connector.connect(host="host.docker.internal",user="admin", password="betersbeters", database="orders")     

c = cnx.cursor()

c.execute("SELECT * FROM market_orders LIMIT 0,5;")

for x in c:
	print(x)

c.close()
cnx.close()
