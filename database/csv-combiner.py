import time
import os
import datetime

#Timestamp format "yyyy-mm-dd hh:mm:ss"
def timestamp_to_epoch(timestamp):
    date_time = '29.08.2011 11:05:02'
    pattern = '%Y-%m-%d %H:%M:%S'
    epoch = int(time.mktime(time.strptime(timestamp, pattern)))
    return epoch

dir_path = "data"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
rootdir = '../data-scraper/data/marketorders'
now = datetime.datetime.now()
combined_csv = open(dir_path + "/" + now.strftime("%Y-%m-%d %H:%M:%S")+".csv", 'w')
combined_csv.write("sep=;\n")
combined_csv.write("stock;sector;publication_time;mmt_flags;transaction_id_code;price;volume\n")

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        csv_file = open(subdir + "/" + file, 'r')
        text = csv_file.readline()
        if text == "sep=;\n": #Check for corruption in start of file
            next(csv_file) #Skip CSV headers
            sector = subdir.split("/")[-1]
            stock_name = file[:-4]
            
            while True:
                line = csv_file.readline()
                
                # if line is empty end of file is reached
                if not line:
                    break
                split = line.split(";", 1)
                pub_epoch = timestamp_to_epoch(split[0])
                combined = stock_name+";"+sector+";"+str(pub_epoch) + ";" + split[1]
                
                combined_csv.write(combined)
        csv_file.close()


