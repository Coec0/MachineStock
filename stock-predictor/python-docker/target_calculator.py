import math
import pandas as pd

class TargetCalculator:

    def __init__(self, stock, dataframe, start_time=1604304002):
        self.stock = stock
        self.dataset = dataframe[(dataframe["publication_time"] >= start_time) & (dataframe["stock"] == stock)]
        self.pointer = 0
        self.current_time = start_time
        self.current_data = None
        self.values = [None for _ in range(21)]
        print(self.dataset)
        self.step() #Step to end of start_time and also set current_data
        
    #Step one second
    def step(self):
        if self.dataset["publication_time"].iloc[self.pointer+1] == self.current_time:
            self.pointer += self.__last_entry_offset_from_timestamp(self.pointer)
            self.current_data = self.dataset.iloc[self.pointer]
        
        self.current_time+=1
        self.values[0] = self.__latest_price_after_time(15)
        self.values[1] = self.__average_price_after_time(15)
        self.values[2] = self.__price_up_down(15)
        self.values[3] = self.__latest_price_after_time(30)
        self.values[4] = self.__average_price_after_time(30)
        self.values[5] = self.__price_up_down(30)
        self.values[6] = self.__latest_price_after_time(45)
        self.values[7] = self.__average_price_after_time(45)
        self.values[8] = self.__price_up_down(45)
        self.values[9] = self.__latest_price_after_time(60)
        self.values[10] = self.__average_price_after_time(60)
        self.values[11] = self.__price_up_down(60)
        self.values[12] = self.__latest_price_after_time(180)
        self.values[13] = self.__average_price_after_time(180)
        self.values[14] = self.__price_up_down(180)
        self.values[15] = self.__latest_price_after_time(300)
        self.values[16] = self.__average_price_after_time(300)
        self.values[17] = self.__price_up_down(300)
        self.values[18] = self.__latest_price_after_time(600)
        self.values[19] = self.__average_price_after_time(600)
        self.values[20] = self.__price_up_down(600)       
        
        
    #Get current second
    def get(self):
        print(self.current_data["price"])
        return self.values
    
    def __last_entry_offset_from_timestamp(self, pointer):
        offset=0
        while self.current_time == self.dataset["publication_time"].iloc[pointer+offset+1]:
            offset+=1 #Loop until pointing at last entry of timestamp
        return offset
        
    def __average_price_after_time(self, time):
        beta = 0.05
        limit = math.ceil(beta * time)
        upper_limit = self.dataset["publication_time"] <= (self.current_time + time + limit)
        lower_limit = self.dataset["publication_time"] >= (self.current_time + time - limit)
        prices = self.dataset[lower_limit & upper_limit]["price"]
        return self.__latest_price_after_time(time) if prices.size == 0 else prices.agg(['mean']).iloc[0]
        
        
    
    def __latest_price_after_time(self, time):
        lookup_time = self.current_time + time
        return self.dataset[self.dataset["publication_time"] <= lookup_time].iloc[-1]["price"]
        
    def __price_up_down(self, time):
        price = self.__latest_price_after_time(time)
        if price < self.current_data["price"]:
            return -1
        if price > self.current_data["price"]:
            return 1
        return 0
        
        
        
        
        
        
        
        
        
        
        
    
