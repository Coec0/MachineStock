import math
from collections import deque

class PriceChannels:
    def __init__(self, time_window, sections, normalize):
        self.time_window = time_window
        self.normalize = normalize
        self.orders = deque(maxlen=time_window)
        self.section = time_window // sections
        self.max_line = (0,0) #k,m
        self.min_line = (0,0) #k,m
        
    def update(self, mo): #Dict
        if len(self.orders) == 0 or self.orders[0]["publication_time"] != mo["publication_time"]:
            self.orders.append(mo)
        else:
            self.orders.pop()
            self.orders.append(mo)
        max1, max2 = self.upper_channel()
        min1, min2 = self.lower_channel()
        k_max = self.calc_k(max1, max2, self.normalize)
        k_min = self.calc_k(min1, min2, self.normalize)
        m_max = self.calc_m(max1, k_max)
        m_min = self.calc_m(min1, k_min)
        self.max_line = (k_max, m_max)
        self.min_line = (k_min, m_min)
        
    def get_min_max_k(self):
        if self.normalize:
            return math.atan(self.max_line[0])/(math.pi/2), math.atan(self.min_line[0])/(math.pi/2)
        return self.min_line[0], self.max_line[0]
        
    def get_relativity_in_price_channel(self):
        latest_order = self.orders[0]["price"]
        y_max = __calc_y(self.max_line[0], latest_order, self.max_line[1])
        y_min = __calc_y(self.min_line[0], latest_order, self.min_line[1])
            
        return (latest_order-y_min)/(y_max-y_min)
        
    def __calc_y(self, k, x, m):
        return k*x+m
        
    def calc_k(self, point1, point2, normalize):
        if point2[0] - point1[0] == 0:
            return 0
        return (point2[1] - point1[1])/(point2[0] - point1[0])
        
    def calc_m(self, point, k):
        return point[1] - k*point[0]
        
    def upper_channel(self):
        max_value1_point = (0,0) #Price, time
        for count, order in enumerate(self.orders):
            if order["price"] > max_value1_point[0]:
                max_value1_point = (order["price"], count)
            if count > self.section: #Loop until one section has been reached
                break
        
        max_value2_point = (0,0) #Price, time
        for count, order in enumerate(reversed(self.orders)):
            if order["price"] > max_value2_point[0]:
                max_value2_point = (order["price"], len(self.orders) - count)
            if count > self.section: #Loop until one section has been reached
                break
        
        return max_value1_point, max_value2_point
        
    def lower_channel(self):
        min_value1_point = (10000000,0) #Price, time
        for count, order in enumerate(self.orders):
            if order["price"] < min_value1_point[0]:
                min_value1_point = (order["price"], count)
            if count > self.section: #Loop until one section has been reached
                break
        
        min_value2_point = (10000000,0) #Price, time
        for count, order in enumerate(reversed(self.orders)):
            if order["price"] < min_value2_point[0]:
                min_value2_point = (order["price"], len(self.orders) - count)
            if count > self.section: #Loop until one section has been reached
                break
        
        return min_value1_point, min_value2_point
            
