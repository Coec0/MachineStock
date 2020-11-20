from request_omx import RequestOMX

stock_id = "SSE365"

fetcher = RequestOMX()

x = fetcher.fetch(stock_id, "2020-11-18")

outF = open("test.csv", "w")
outF.write(x.text)
