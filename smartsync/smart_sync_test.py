from smart_sync import SmartSync


#  Test everything into a single row
def test1():
    ts1 = 1
    smart = SmartSync(3, 3)
    if smart.put(ts1, 0, 1) is not None:
        print("ERROR Test1, 1")
    if smart.put(ts1, 1, 2) is not None:
        print("ERROR Test1, 2")
    if smart.put(ts1, 2, 3) is None:
        print("ERROR Test1, 3")


#  Test Two rows almost filled, then filling them
def test2():
    ts1 = 1
    ts2 = 2
    smart = SmartSync(3, 3)
    if smart.put(ts1, 0, 1) is not None:
        print("ERROR Test2, 1")
    if smart.put(ts1, 1, 2) is not None:
        print("ERROR Test2, 2")

    if smart.put(ts2, 0, 1) is not None:
        print("ERROR Test2, 3")
    if smart.put(ts2, 1, 2) is not None:
        print("ERROR Test2, 4")

    if smart.put(ts1, 2, 3) is None:
        print("ERROR Test2, 5")
    if smart.put(ts2, 2, 3) is None:
        print("ERROR Test2, 6")


#  Test cycling the same row
def test3():
    ts1 = 1
    ts2 = 4  # 4 mod 3 = 1
    smart = SmartSync(3, 3)
    if smart.put(ts1, 0, 1) is not None:
        print("ERROR Test3, 1")
    if smart.put(ts1, 1, 2) is not None:
        print("ERROR Test3, 2")

    if smart.put(ts2, 0, 1) is not None:
        print("ERROR Test3, 3")
    if smart.put(ts2, 1, 2) is not None:
        print("ERROR Test3, 4")
    if smart.put(ts2, 2, 3) is None:
        print("ERROR Test3, 5")


#  Test cycling the same row but different order
def test4():
    ts1 = 1
    ts2 = 4  # 4 mod 3 = 1
    smart = SmartSync(3, 3)
    if smart.put(ts1, 0, 1) is not None:
        print("ERROR Test4, 1")
    if smart.put(ts1, 1, 2) is not None:
        print("ERROR Test4, 2")

    if smart.put(ts2, 2, 1) is not None:
        print("ERROR Test4, 3")
    if smart.put(ts2, 1, 2) is not None:
        print("ERROR Test4, 4")
    if smart.put(ts2, 0, 3) is None:
        print("ERROR Test4, 5")


#  Fill twice cycle
def test5():
    ts1 = 1
    ts2 = 4  # 4 mod 3 = 1
    smart = SmartSync(3, 3)
    if smart.put(ts1, 0, 1) is not None:
        print("ERROR Test5, 1")
    if smart.put(ts1, 1, 2) is not None:
        print("ERROR Test5, 2")
    if smart.put(ts1, 2, 3) is None:
        print("ERROR Test5, 3")

    if smart.put(ts2, 0, 1) is not None:
        print("ERROR Test5, 1")
    if smart.put(ts2, 1, 2) is not None:
        print("ERROR Test5, 2")
    if smart.put(ts2, 2, 3) is None:
        print("ERROR Test5, 3")


#  Insert old cycle into new cycle
def test6():
    ts1 = 1
    ts2 = 4  # 4 mod 3 = 1
    smart = SmartSync(3, 3)
    if smart.put(ts2, 0, 1) is not None:
        print("ERROR Test6, 1")
    if smart.put(ts2, 1, 2) is not None:
        print("ERROR Test6, 2")
    if smart.put(ts1, 2, 3) is not None:
        print("ERROR Test6, 3")
    if smart.put(ts2, 2, 3) is None:
        print("ERROR Test6, 4")


test1()
test2()
test3()
test4()
test5()
test6()
