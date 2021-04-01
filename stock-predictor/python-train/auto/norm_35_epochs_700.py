import trainbase
import itertools
import main

models = [trainbase.DeepModel()] #trainbase.ShallowModel()
window_sizes = [700]
fin_inds = ["price", "ema", "rsi", "macd", "volatility", "channels"]
cols_y = [("5s", 0), ("30s", 4)]
epochs = [35]
batch_sizes = [512]
learning_rates = [0.0001, 0.000001]
useTime = [True, False]
normal = [True]
name = ["norm_35_epochs"]

iterator = itertools.product(models, epochs, batch_sizes, window_sizes, fin_inds, learning_rates, cols_y, useTime, normal, name)

main.run(iterator)
