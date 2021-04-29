import trainbase
import itertools
import main

models = [trainbase.DeepModel()]
window_sizes = [200, 700]
fin_inds = ["price"]
cols_y = [("30s", 4)]
epochs = [5]
batch_sizes = [1024]
learning_rates = [0.00001]
useTime = [False]
normal = [True]
name = ["nordea_200_700_auto"]

iterator = itertools.product(models, epochs, batch_sizes, window_sizes, fin_inds, learning_rates, cols_y, useTime, normal, name)

main.run(iterator)
