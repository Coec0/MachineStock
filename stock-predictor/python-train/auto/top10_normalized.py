import trainbase
import itertools
import main

combinations = [[trainbase.DeepModel(), 35, 512, 700, "macd", 0.000001, ("30s", 0), False, True, "top_10_normalized"],
                [trainbase.DeepModel(), 35, 512, 700, "channels", 0.000001, ("30s", 0), False, True, "top_10_normalized"],
                [trainbase.DeepModel(), 5, 512, 70, "channels", 0.0001, ("5s", 0), True, True, "top_10_normalized"],
                [trainbase.DeepModel(), 35, 512, 200, "ema", 0.000001, ("30s", 0), True, True, "top_10_normalized"],
                [trainbase.DeepModel(), 5, 512, 200, "rsi", 0.0001, ("5s", 0), False, True, "top_10_normalized"],
                [trainbase.DeepModel(), 5, 512, 200, "ema", 0.000001, ("15s", 0), False, True, "top_10_normalized"],
                [trainbase.DeepModel(), 5, 512, 200, "price", 0.000001, ("30s", 0), False, True, "top_10_normalized"],
                [trainbase.DeepModel(), 5, 512, 200, "price", 0.000001, ("15s", 0), True, True, "top_10_normalized"],
                [trainbase.DeepModel(), 35, 512, 70, "rsi", 0.000001, ("30s", 0), True, True, "top_10_normalized"],
                [trainbase.DeepModel(), 35, 512, 70, "ema", 0.000001, ("30s", 0), True, True, "top_10_normalized"]]
iterator = iter(combinations)


main.run(iterator)
