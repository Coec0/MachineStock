import trainbase
import itertools
import main

models = [trainbase.DeepModel()] #trainbase.ShallowModel()
window_sizes = [70, 200, 700]
fin_inds = ["price", "ema", "rsi", "macd", "volatility", "channels"]
cols_y = [("5s", 0), ("15s", 2), ("30s", 4), ("60s", 6)]
epochs = [5]
batch_sizes = [512]
learning_rates = [0.01, 0.0001, 0.000001]
useTime = [False]
normal = [True]

iterator = itertools.product(models, epochs, batch_sizes, window_sizes, fin_inds, learning_rates, cols_y, useTime, normal)

main.run(iterator)
