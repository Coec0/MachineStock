from urllib.parse import quote
import requests
import random
import string
import time

class RequestOMX:

    def __init__(self, cookie_id="54EBA48763AFA30638BE5DE8B969E05E"):
        self.cookie_id = cookie_id

    def change_cookie_id(self):
        index = random.randint(1, len(self.cookie_id)-2)
        rand_char = random.choice(string.ascii_uppercase + string.digits)
        self.cookie_id = self.cookie_id[:index-1] + rand_char + self.cookie_id[index:]

    def fetch(self, stock_id, f_date, t_date = ''):
        response = self.create_and_post_request(stock_id, f_date, t_date)
        self.print_response_message(response, stock_id)
        timelimit = 60 - response.elapsed_time

        while timelimit > 0 and (response.status_code != 200 or len(response.content)//1000 == 0):
            self.change_cookie_id()
            response = self.create_and_post_request(stock_id, f_date, t_date)
            self.print_response_message(response, stock_id)
            timelimit = 60 - response.elapsed_time

        if(timelimit <= 0):
            print("Timeout ("+stock_id+")")

        return response

    def create_and_post_request(self, stock_id, f_date, t_date = ''):
        from_date = '<param name="FromDate" value="'+f_date+'"/>\n'
        to_date = '' if t_date == '' else '<param name="ToDate" value="'+t_date+'"/>'
        instrument = '<param name="Instrument" value="'+ stock_id +'"/>'
        url = "http://www.nasdaqomxnordic.com/webproxy/DataFeedProxy.aspx"

        xmlquery = '''<post>
        <param name="SubSystem" value="Prices"/>
        <param name="Action" value="GetTrades"/>
        <param name="Exchange" value="NMF"/>
        <param name="t__a" value="32,28,99,1,2"/>
        ''' + from_date + to_date + instrument +'''
        <param name="ext_contenttype" value="application/ms-excel"/>
        <param name="ext_contenttypefilename" value="share_export.csv"/>
        <param name="ext_xslt" value="/nordicV3/trades_csv.xsl"/>
        <param name="ext_xslt_lang" value="en"/>
        <param name="showall" value="1"/>
        <param name="app" value="/shares/microsite"/>
        </post>'''
        xmlquery = "xmlquery="+quote(xmlquery, safe='')

        headers = {
         "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:82.0) Gecko/20100101 Firefox/82.0",
         "Accept" : "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
         "Accept-Language" : 'en-US,en;q=0.5',
         "Content-Type": "application/x-www-form-urlencoded",
         "Origin" : "http://www.nasdaqomxnordic.com",
         "Connection" : "keep-alive",
         "Referer" : "http://www.nasdaqomxnordic.com/shares/microsite?Instrument="+stock_id,
         "Cookie" : "JSESSIONID="+self.cookie_id,
         "Upgrade-Insecure-Requests" : "1"
        }

        cookies = {
         "NSC_MC_Obtebrpnyopsejd_IUUQ" : "ffffffff09be0e1e45525d5f4f58455e445a4a423660",
         "ASP.NET_SessionId":"btzdum55rggcgvf55yx4xe45",
         "nasdaqomxnordic_sharesform" : 'radio"%"3DnordicShares"%"26market"%"3D0"%"26balticMarket"%"3D0"%"26segment"%"3D0"%"26fnSegment"%"3D0',
         "ak_bmsc" : "3245F4E1095A5A43FEAB27380A86DC8B17411D5FE23A00008CD0B65FE44B036A~plpePFz8CPhibYx7B8tBtAy/NOjbdmJZGLppqCNLPa7Gi6UuXuxZ4+zrcRL5ZNs+jLIDlq2gG0or6hzIUU6JssmPLIqDL9cAKrk8/weRKvv9SWZ/7bpAW5Vx5i+/Toz87Qj53CBfgmzCaCNXhQ4o3YkO9HcGtrLDZ33mDeGJ2dsBaJe8Kp2asVuE3K5bgqfCgQDNoYrRHfEDE8BQTWwnNeRsf92lyJJ9M/PUGdHvogYqihLxNW/jAHXwEzfZgdTKx+",
         "bm_sv" : "70E2D1844C3EBE9B4557677EB0C5E9BC~fBsyTAML3FSFIcgiDtEghR4lGsQ3bM/c3bUAVJTxMJIPojR3vtrnNO4QNIT3p/XTl7SLshTQbXZQC5nxpa3YHuDeNaGaZMkON+oUZseMzW60mGznpjyKW0yMawlzxRPM4yPZPV4miWMyq67cj6r8TK986YnPsneMbbwmJP8UPfg=",
         "bm_mi":"AF983259C704372E590D65A4FEE8D234~v/FwEZugv2ZCHTw1TjO1qhp9mN1zYAvJp1uT0cXQAYIFvmd1DSQhoe9LF2RyjYg6ciIOsNW49Fi/xDYEGEdw+u2YTCGh659UBP41wYyhivbTngS9DBUFZndSqJpop/yOsbvkckgHhUs4iItcf9csUXZq4Jbt9rYdXUrtNYkFwSmFgaQDts6M2w5HQdI/W4awCDa+lbYQ7R1k3IDBPexiu7gP1XsfNMPjmESv5lkd4rf4GlWgjbpr3n0vjwajDVhi"
        }

        start = time.time()
        res = requests.post(url, data = xmlquery, cookies = cookies, headers = headers)
        roundtime = time.time() - start
        res.elapsed_time = roundtime
        return res

    def print_response_message(self, res, stock_id):
        msg = "Response: [{0}], Size: {1}KB, Time: {2}ms ({3})"
        msg = msg.format(res.status_code, len(res.content)//1000, round(res.elapsed_time*1000), stock_id)
        print(msg)
