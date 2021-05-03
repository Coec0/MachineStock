import logging
import time

from smart_sync import SmartSync
from smart_sync_old import SmartSyncOld


def bench(ws, number_of_nodes):
    sync = SmartSync(ws, number_of_nodes)
    start_time = time.time()
    for i in range(number_of_nodes):
        sync.put(1, i, 10)
    return time.time() - start_time


def bench_old(ws, number_of_nodes):
    sync = SmartSyncOld(ws, number_of_nodes)
    start_time = time.time()
    for i in range(number_of_nodes):
        sync.put(1, i, 10)
    return time.time() - start_time


print("NEW")
for j in range(6):
    print("input:"+str(pow(10, j+1)))
    avg = 0
    for k in range(10):
        run = bench(10, pow(10, j+1))
        print("Run "+str(k)+": {:.10f}".format(run))
        avg += run/10
    print("Average: {:.10f}".format(avg)+"\n")

print("OLD")
for j in range(6):
    print("input:"+str(pow(10, j+1)))
    avg = 0
    for k in range(10):
        run = bench_old(10, pow(10, j+1))
        print("Run "+str(k)+": {:.10f}".format(run))
        avg += run/10
    print("Average: {:.10f}".format(avg)+"\n")

