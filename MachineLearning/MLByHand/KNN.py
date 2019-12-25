import numpy as np
from collections import defaultdict

def eclid_distance(x1, x2):

    return np.sqrt(np.sum((x1_i-x2_i)**2 for x1_i, x2_i in zip(x1, x2)))


class KNN:
    def __init__(self):
        pass

    def get_neighbors(self, train_x, train_label, test_row, num_neighbors):
        distances = list()
        for train_row, train_label_row in zip(train_x, train_label):
            dist = eclid_distance(test_row, train_row)
            distances.append((dist, train_label_row))
        distances.sort(key=lambda tup: tup[0])
        return distances[:num_neighbors]

    def vote(self, train_x, train_label, test_row, num_neighbors):
        distances = self.get_neighbors(train_x, train_label, test_row, num_neighbors)

        distances = [(1/dist, train_label_row) for dist, train_label_row in distances]
        _sum = sum(distance[0] for distance in distances)
        distances = [(dist/_sum, train_label_row) for dist, train_label_row in distances]
        votes = defaultdict(lambda: 0)
        for inv_dist, train_label_row in distances:
            votes[train_label_row] += inv_dist

        predicted = max(votes, key=lambda vote_i: votes[vote_i])

        return votes







if __name__=="__main__":
    data = np.array([
        [0, 0.32, 0.43, 0], [1, 0.26, 0.54, 0],
        [2, 0.27, 0.60, 0], [3, 0.37, 0.36, 0],
        [4, 0.37, 0.68, 0], [5, 0.49, 0.32, 0],
        [6, 0.46, 0.70, 0], [7, 0.55, 0.32, 0],
        [8, 0.57, 0.71, 0], [9, 0.61, 0.42, 0],
        [10, 0.63, 0.51, 0], [11, 0.62, 0.63, 0],
        [12, 0.39, 0.43, 1], [13, 0.35, 0.51, 1],
        [14, 0.39, 0.63, 1], [15, 0.47, 0.40, 1],
        [16, 0.48, 0.50, 1], [17, 0.45, 0.61, 1],
        [18, 0.55, 0.41, 1], [19, 0.57, 0.53, 1],
        [20, 0.56, 0.62, 1], [21, 0.28, 0.12, 1],
        [22, 0.31, 0.24, 1], [23, 0.22, 0.30, 1],
        [24, 0.38, 0.14, 1], [25, 0.58, 0.13, 2],
        [26, 0.57, 0.19, 2], [27, 0.66, 0.14, 2],
        [28, 0.64, 0.24, 2], [29, 0.71, 0.22, 2]],
        dtype=np.float32)
    x_train = np.array([data_i[1:-1] for data_i in data])
    y_train = [data_i[-1] for data_i in data]
    model = KNN()

    print(model.vote(x_train, y_train, [0.62, 0.35], 10))


