#!/usr/bin/env python3

import time

import pydtnn

# Configuration
SERVER_IP = "127.0.0.1"
SERVER_PORT = 6526
DEVICE_NAME = "Jetson-Xavier"
SECONDS_TO_SLEEP = 2

print("Set server and lines...")
pmlib = pydtnn.power_meters.PMLib(SERVER_IP, SERVER_PORT)
#
print(f"{pmlib.server.server_ip=}")
print(f"{pmlib.server.port=}")
#
print(f"{pmlib.lines.__bits=} ({int.from_bytes(pmlib.lines.__bits, byteorder='little'):032b})")
print()

print("Creating counter...")
pmlib.create_counter(DEVICE_NAME, 1, 0)
print()

print(f"Starting counter at timestamp {time.time()}...")
pmlib.start_counter()
print()

print(f"Sleeping {SECONDS_TO_SLEEP} second...")
time.sleep(SECONDS_TO_SLEEP)
print()

print(f"Stopping counter at timestamp {time.time()}...")
pmlib.stop_counter()
print()

print("Getting data...")
pmlib.get_counter_data()

print(f"{pmlib.counter.measures.contents.next_timing=}")
print(f"{pmlib.counter.measures.contents.timing=}")
print(f"{pmlib.counter.measures.contents.energy.watts_size=}")
print(f"{pmlib.counter.measures.contents.energy.watts_sets_size=}")
print(f"{pmlib.counter.measures.contents.energy.watts_sets=}")
print(f"{pmlib.counter.measures.contents.energy.watts=}")
print(f"{pmlib.counter.measures.contents.energy.lines_len=}")
print()
print(f"{pmlib.times=}")
print(f"{pmlib.watts=}")
print()

print("Storing output on 'pmlib_example_output.txt'")
pmlib.print_data_text("pmlib_example_output.txt", -1)
print()


def print_joules(start_time, end_time):
    print(f"Computing joules between {start_time:f} and {end_time:f}...")
    joules = pmlib.get_joules(start_time, end_time, debug=True)
    print(f"Joules = {joules}")
    print(f"Number of intermediate samples = {pmlib.get_number_of_intermediate_samples(start_time, end_time)}")
    print()


for pairs in [
    (pmlib.times[0], pmlib.times[-1]),
    (pmlib.times[0] + 0.25, pmlib.times[0] + 0.3)
]:
    print_joules(pairs[0], pairs[1])

# Must be done AFTER integrating the energy (the watts array will be lost if not explicitly copied)
print("Finalize counter...")
pmlib.finalize_counter()
print()
