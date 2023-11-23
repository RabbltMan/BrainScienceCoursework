from collections import Counter
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential, layers
from keras.optimizers import Adam
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


class P300:
    """
    P300 Signal Recognition
    """

    def __init__(self) -> None:
        self.data = P300.data_split()
        self.net = P300.create_network()
        self.pos_char_map = P300.generate_map()
        self.fit_model()
        # self.generate_plot()
        self.test()

    @staticmethod
    def generate_map():
        """
        Generate a map of (row, col) -> character
        """
        pos_char_map = {}
        for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"):
            pos_char_map[(i // 6 + 1, i % 6 + 7)] = c
        return pos_char_map

    @staticmethod
    def data_split(test_size: float = 0.05) -> Tuple[np.ndarray]:
        """
        Load preprocessed data from file and split into 
        `(X_train, X_val, y_train, t_test)` subsets.
        """
        X_train = np.load("./P300/X_train.npy")
        y_train = np.load("./P300/y_train.npy")
        return train_test_split(X_train, y_train, test_size=test_size)

    @staticmethod
    def create_network():
        """
        Create a model to process signal data
        """
        net = Sequential()
        net.add(layers.Conv1D(20, 5))
        net.add(layers.BatchNormalization())
        net.add(layers.MaxPooling1D())
        net.add(layers.Conv1D(40, 3))
        net.add(layers.BatchNormalization())
        net.add(layers.MaxPooling1D())
        net.add(layers.Conv1D(50, 2))
        net.add(layers.BatchNormalization())
        net.add(layers.Flatten())
        net.add(layers.Dense(256))
        net.add(layers.Dense(2, activation="softmax"))
        optim = Adam(0.00015)
        net.compile(optim,
                    loss='categorical_crossentropy',
                    metrics=['categorical_accuracy'])
        return net

    def fit_model(self):
        """
        method which trains the model
        """
        X, y = self.data[0], self.data[2]
        X_val, y_val = self.data[1], self.data[3]
        class_counts = Counter(np.argmax(y, axis=1))
        class_weights = {
            i: len(y) / class_counts[i]
            for i in range(len(class_counts))
        }
        self.history = self.net.fit(x=X,
                                    y=y,
                                    batch_size=64,
                                    epochs=50,
                                    class_weight=class_weights)
        self.svm = SVC(C=7.0, kernel='rbf')
        self.svm.fit(X.reshape(X.shape[0], -1), np.argmax(y, axis=1))
        y_pred = self.svm.predict(X_val.reshape(X_val.shape[0], -1))
        score = f1_score(np.argmax(y_val, axis=1), y_pred)
        print("Validation score:", score)

    def generate_plot(self):
        """
        Generate loss and categorical accuracy plot
        """
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(self.history.history['loss'],
                 color='red',
                 marker='x',
                 label='Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss', color='red')
        ax1.tick_params(axis='y', colors='red')
        ax2.plot(self.history.history['categorical_accuracy'],
                 color='blue',
                 marker='x',
                 label='Categorical Accuracy')
        ax2.set_ylabel('Categorical Accuracy', color='blue')
        ax2.tick_params(axis='y', colors='blue')
        fig.tight_layout()
        plt.savefig('./P300/figure.png')

    def test(self):
        """
        test model performance on test cases
        """
        X_test: np.ndarray = np.load("./P300/X_test.npy")
        y_pred = self.svm.predict(X_test.reshape(X_test.shape[0], -1))
        row, group, ans = [], [], []
        for label in y_pred:
            row.append(label)
            if len(row) == 12:
                group.append(np.array(row))
                row = []
            if len(group) == 5:
                # print(np.array(group))
                output = np.sum(np.array(group), axis=0)
                print(output, end=' ')
                row_max = np.max(output[:6])
                col_max = np.max(output[6:])
                row_max_idx = np.where(output[:6] == row_max)[0] + 1
                col_max_idx = np.where(output[6:] == col_max)[0] + 7
                char_options = []
                for i in row_max_idx:
                    for j in col_max_idx:
                        char_options.append(self.pos_char_map[(i, j)])
                print(f"-> {char_options}")
                ans.append(char_options)
                group = []
        print()
        y_pred = self.net(X_test)
        row, group = [], []
        for label in y_pred:
            row.append(np.argmax(label.numpy()))
            if len(row) == 12:
                group.append(np.array(row))
                row = []
            if len(group) == 5:
                # print(np.array(group))
                output = np.sum(np.array(group), axis=0)
                print(output, end=' ')
                row_max = np.max(output[:6])
                col_max = np.max(output[6:])
                row_max_idx = np.where(output[:6] == row_max)[0] + 1
                col_max_idx = np.where(output[6:] == col_max)[0] + 7
                char_options = []
                for i in row_max_idx:
                    for j in col_max_idx:
                        char_options.append(self.pos_char_map[(i, j)])
                print(f"-> {char_options}")
                group = []


P300()
