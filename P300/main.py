import numpy as np
from pandas import read_excel

DATA_RANGE = "BDGLOQSVZ479"
char_pos_map = {}
pos_char_map = {}
for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"):
    if c in DATA_RANGE:
        char_pos_map[c] = (i // 6 + 1, i % 6 + 7)
        pos_char_map[(i // 6 + 1, i % 6 + 7)] = c

X_train = [None for _ in range(len(DATA_RANGE) * 12 * 5)]
y_train = 
for i, c in enumerate(DATA_RANGE):
    data = read_excel("./P300/S5/S5_train_data.xlsx",
                      sheet_name=i,
                      header=None)
    data = (data - data.min()) / (data.max() - data.min())
    data = data.to_numpy()
    event = read_excel("./P300/S5/S5_train_event.xlsx",
                       sheet_name=i,
                       header=None)
    event_cond = event[0] < 100
    event = event[event_cond].to_numpy()
    
