import queue

class DataProcessor:
    def __init__(self, stock, window_size, usePrice=True, beta=0.98, useVol=True ,useExpAvgPrice=True): #["price, volume, --mnt--p, publicationtime"]
        self.useExpAvgPrice = useExpAvgPrice
        self.usePrice = usePrice
        self.useVol = useVol
        self.stock = stock
        self.beta = beta
        self.window_size = window_size
        self.processed = {"window" : queue.Queue(maxsize=window_size),
                          "exp_avg_price" : 0}

    def process(self, market_order): #JSON/dict
        mr = []
        if(self.usePrice):
            mr.append(market_order["price"])
        if(self.useVol):
            mr.append(market_order["volume"])

        self.processed["window"].append(mr)

        if(self.useExpAvgPrice):
            self.update_exp_avg_price(market_order["price"])

    def get_processed(self):
        return self.processed

    def update_exp_avg_price(self, price):
        old_exp_avg = self.processed["exp_avg_price"]
        new_exp_avg = self.beta*old_exp_avg + (1-self.beta)*price
        self.processed["exp_avg_price"] = new_exp_avg
