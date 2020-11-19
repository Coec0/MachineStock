import request_omx

stock_id = "SSE365"

x = request_omx.fetch(stock_id, "2020-11-18")

outF = open("test.csv", "w")
outF.write(x.text)

#print(x.text)
