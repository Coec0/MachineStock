print("hejhejhej")
import torch
x = torch.rand(5, 3)
print(x)

import mysql.connector

cnx = mysql.connector.connect(host="host.docker.internal",user="admin", password="kvxxkv11-sql", database="test")     

c = cnx.cursor()

c.execute("SELECT * FROM A")

for x in c:
	print(x[0])

c.close()
cnx.close()   

