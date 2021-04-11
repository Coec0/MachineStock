import trainbase
import itertools
import main

combinations = [[trainbase.DeepModel(), 35, 512, 70, "price", 0.000001, ("5s", 0), False, False, "top_10"],
                [trainbase.DeepModel(), 35, 512, 70, "ema", 0.000001, ("5s", 0), False, False, "top_10"],
                [trainbase.DeepModel(), 35, 512, 70, "price", 0.0001, ("5s", 0), False, False, "top_10"],
                [trainbase.DeepModel(), 5, 512, 70, "channels", 0.000001, ("5s", 0), False, False, "top_10"],
                [trainbase.DeepModel(), 5, 512, 70, "ema", 0.000001, ("5s", 0), False, False, "top_10"],
                [trainbase.DeepModel(), 35, 512, 70, "price", 0.000001, ("30s", 0), False, False, "top_10"],
                [trainbase.DeepModel(), 5, 512, 70, "price", 0.000001, ("5s", 0), False, False, "top_10"],
                [trainbase.DeepModel(), 5, 512, 70, "macd", 0.000001, ("5s", 0), False, False, "top_10"],
                [trainbase.DeepModel(), 5, 512, 70, "volatility", 0.000001, ("5s", 0), False, False, "top_10"],
                [trainbase.DeepModel(), 35, 512, 70, "ema", 0.000001, ("30s", 0), False, False, "top_10"]]
iterator = iter(combinations)


main.run(iterator)
