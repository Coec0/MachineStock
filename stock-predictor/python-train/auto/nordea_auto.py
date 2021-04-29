import trainbase
import itertools
import main

models = [trainbase.DeepModel()]
window_sizes = [70]
fin_inds = ["price"]
cols_y = [("30s", 4)]
epochs = [60, 65, 70]
batch_sizes = [512]
learning_rates = [0.00001]
useTime = [True]
normal = [True]
name = ["nordea_70_auto"]

iterator = itertools.product(models, epochs, batch_sizes, window_sizes, fin_inds, learning_rates, cols_y, useTime, normal, name)

main.run(iterator)
