from urllib.parse import quote
import requests

def fetch(stock_id, f_date, t_date = '') :
    cookie_id = "54EBA48763AFA30638BE5DE8B969E05E"

    from_date = '<param name="FromDate" value="'+f_date+'"/>\n'
    to_date = '' if t_date == '' else '<param name="ToDate" value="'+t_date+'"/>'
    instrument = '<param name="Instrument" value="'+ stock_id +'"/>'

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

    xmlquery = quote(xmlquery, safe='')

    url = "http://www.nasdaqomxnordic.com/webproxy/DataFeedProxy.aspx"

    headers = {
     "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:82.0) Gecko/20100101 Firefox/82.0",
     "Accept" : "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
     "Accept-Language" : 'en-US,en;q=0.5',
     "Content-Type": "application/x-www-form-urlencoded",
     "Origin" : "http://www.nasdaqomxnordic.com",
     "Connection" : "keep-alive",
     "Referer" : "http://www.nasdaqomxnordic.com/shares/microsite?Instrument="+stock_id,
     "Cookie" : "JSESSIONID="+cookie_id,
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

    return requests.post(url, data = "xmlquery="+xmlquery, cookies = cookies, headers = headers)
