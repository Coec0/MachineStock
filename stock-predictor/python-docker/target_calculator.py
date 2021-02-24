import math
import pandas as pd
from threading import Thread
import subprocess
from datetime import datetime
from datetime import timedelta

class TargetCalculator:

    def __init__(self, stock, dataframe, start_time=1604358000):#1604390402): #1604304002
        self.stock = stock
        self.dataset = dataframe[(dataframe["publication_time"] >= start_time) & (dataframe["stock"] == stock)].reset_index(drop=True)
        self.dataset
        self.pointer = 0
        self.current_time = start_time
        self.current_price = dataframe[(dataframe["publication_time"] <= start_time) & (dataframe["stock"] == stock)].iloc[-1]["price"]
        self.values = [None for _ in range(21)]
        print(self.dataset)
        self.step() #Step to end of start_time and also set current_price
        self.__update_get_data()
        
    #Step one second
    def step(self):
        if self.pointer+1 >= len(self.dataset):
            return False
            
        if self.dataset.at[self.pointer+1, "publication_time"] <= self.current_time:
            self.pointer += self.__last_entry_offset_from_timestamp(self.pointer)
            self.current_price = self.dataset.at[self.pointer, "price"]
            self.__update_get_data()
        self.current_time+=self.__market_open_time_increase(self.current_time)      
        return True
        
    
    def __update_get_data(self):
        values = [None for _ in range(21)]
        #t1 = Thread(target=self.__update_value_thread, args=(values, 0, 15))
        #t2 = Thread(target=self.__update_value_thread, args=(values, 3, 30))
        #t3 = Thread(target=self.__update_value_thread, args=(values, 6, 45))
        #t4 = Thread(target=self.__update_value_thread, args=(values, 9, 60))
        #t5 = Thread(target=self.__update_value_thread, args=(values, 12, 180))
        #t6 = Thread(target=self.__update_value_thread, args=(values, 15, 300))
        #t7 = Thread(target=self.__update_value_thread, args=(values, 18, 600))

        #t1.start()
        #t2.start()
        #t3.start()
        #t4.start()
        #t5.start()
        #t6.start()
        #t7.start()

        #t1.join()
        #t2.join()
        #t3.join()
        #t4.join()
        #t5.join()
        #t6.join()
        #t7.join()
        
        self.__update_value_thread(values, 0, 15)
        self.__update_value_thread(values, 3, 30)
        self.__update_value_thread(values, 6, 45)
        self.__update_value_thread(values, 9, 60)
        self.__update_value_thread(values, 12, 180)
        self.__update_value_thread(values, 15, 300)
        self.__update_value_thread(values, 18, 600)
        self.values = values
        
    def __update_value_thread(self, value_list, start_index, time):
        future_pointer, value_list[start_index] = self.__latest_price_after_time(time)
        value_list[start_index+1] = self.__average_price_after_time(time, future_pointer)
        value_list[start_index+2] = self.__price_up_down(time, value_list[start_index])
    
    def __market_open_time_increase(self, time):
        dt = datetime.fromtimestamp(time)
        is_open = dt.weekday() <= 4
        is_open = is_open and (dt.hour >= 9 and dt.hour < 18 and (False if dt.hour == 17 and dt.minute >= 29 and dt.second >= 59 else True))
        if is_open:
            return 1
            
        delta_days=0 if dt.hour < 9 else 1
        if dt.weekday() >= 4:
            delta_days = 7 - dt.weekday();
            
        tomorrow = dt.replace(hour=9, minute=0, second=0, microsecond=0) + \
                   timedelta(days=delta_days)
        return (tomorrow - dt).total_seconds()
    
    #Get current second
    def get(self):
        values = self.values.copy()
        values.append(self.current_time)
        return values
    
    def __last_entry_offset_from_timestamp(self, pointer):
        offset=0
        try:
            while self.current_time >= self.dataset.at[pointer+offset+1, "publication_time"]:
                offset+=1 #Loop until pointing at last entry of timestamp
        except:
            print("EOF reached")
        return offset
        
    def __average_price_after_time(self, time, future_pointer):
        beta = 0.05
        limit = math.ceil(beta * time)
        upper_limit_pointer = future_pointer
        lower_limit_pointer = future_pointer
        try:
            while self.dataset.at[lower_limit_pointer, "publication_time"] <= self.current_time+time-limit :
                lower_limit_pointer-=1
        except:
            lower_limit_pointer+=1 #SOF reached
        
        try:
            while self.dataset.at[upper_limit_pointer, "publication_time"] <= self.current_time+time+limit :
                upper_limit_pointer+=1
        except:
            upper_limit_pointer-=1 #EOF reached
        
        if upper_limit_pointer == upper_limit_pointer :
            return self.dataset.at[future_pointer, "price"]
        
        orders = upper_limit_pointer - lower_limit_pointer
        sum_orders=0
        for i in range(orders):
            sum_orders+=self.dataset.at[lower_limit_pointer+i, "price"]

        return sum_orders/orders
    
    
    def __latest_price_after_time(self, time):
        lookup_time = self.current_time + time
        offset = self.pointer
        try:
            while self.dataset.at[offset, "publication_time"] <= lookup_time :
                offset+=1
        except:
            offset-=1 #EOF reached
        return offset, self.dataset.at[offset,"price"]
        
    def __price_up_down(self, time, future_price):
        if future_price < self.current_price:
            return -1
        if future_price > self.current_price:
            return 1
        return 0
        
        
        
        
        
        
        
        
        
        
        
    
