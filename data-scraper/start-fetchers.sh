#!/bin/bash
python3 fetch_limit_orders_omx.py "stocks1.json" &
python3 fetch_limit_orders_omx.py "stocks2.json" &
wait $!
