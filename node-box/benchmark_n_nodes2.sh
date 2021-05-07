#!/bin/bash
for x in $(seq 1 "$1")
do
  python benchmark_1_node.py "price$x" &
  sleep 1
done


