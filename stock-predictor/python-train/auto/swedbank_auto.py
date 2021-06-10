import trainbase
import itertools
import main

models = [trainbase.DeepModel()]
window_sizes = [(70, True), (200, False), (700, False)] # (70, True),
fin_inds = ["price"]
cols_y = [("30s", 0)]
epochs = [30, 50, 150, 200]
batch_sizes = [512]
learning_rates = [0.0001, 0.00001]
name = ["swedbank_auto"]
stock = "Swedbank_A"

iterator = itertools.product(models, epochs, batch_sizes, window_sizes, fin_inds, learning_rates, cols_y, name)

main.run(stock, iterator)
