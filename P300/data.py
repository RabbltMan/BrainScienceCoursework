import numpy as np
from pandas import read_excel
from sklearn.preprocessing import StandardScaler

DATA_RANGE = "BDGLOQSVZ479"
char_pos_map = {}
for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"):
    if c in DATA_RANGE:
        char_pos_map[c] = (i // 6 + 1, i % 6 + 7)
X_train = []
y_train = []
for i, c in enumerate(DATA_RANGE):
    data = read_excel("./P300/S5/S5_train_data.xlsx",
                      sheet_name=i,
                      header=None)
    data = data.to_numpy()
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    event = read_excel("./P300/S5/S5_train_event.xlsx",
                       sheet_name=i,
                       header=None)
    event_cond = event[0] < 100
    event = event[event_cond].to_numpy()
    for idx, time_step in event:
        if idx in char_pos_map[c]:
            X_train.append(data[time_step + 24:time_step + 125:3])
            X_train.append(data[time_step + 25:time_step + 125:3])
            X_train.append(data[time_step + 26:time_step + 126:3])
            y_train.append(np.array([0, 1]))
            y_train.append(np.array([0, 1]))
            y_train.append(np.array([0, 1]))
        else:
            X_train.append(data[time_step + 24:time_step + 125:3])
            y_train.append(np.array([1, 0]))
X_test = [None for _ in range(10 * 12 * 5)]
for i in range(10):
    data = read_excel("./P300/S5/S5_test_data.xlsx", sheet_name=i, header=None)
    data = data.to_numpy()
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    event = read_excel("./P300/S5/S5_test_event.xlsx",
                       sheet_name=i,
                       header=None)
    event_cond = event[0] < 100
    event = event[event_cond].to_numpy()
    for j, (idx, time_step) in enumerate(event):
        test_case_idx = i * 60 + (j // 12) * 12 + (idx - 1)
        X_test[test_case_idx] = data[time_step + 24:time_step + 125:3]

np.save("./P300/X_train.npy", np.array(X_train))
np.save("./P300/y_train.npy", np.array(y_train))
np.save("./P300/X_test.npy", np.array(X_test))
