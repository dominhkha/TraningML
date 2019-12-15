import tensorflow as tf
import numpy as np
from collections import defaultdict,Counter


class NaiveBayes:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def count(self):
        counts = defaultdict(lambda: [1]*len(set(y)))
        for index, x_i in enumerate(self.x):
            for x_i_j in x_i:
                counts[x_i_j][self.y[index]] += 1
        return counts

    def conditional_probabilites(self):
        counts = self.count()
        probability = defaultdict(lambda: [0]*len(set(y)))
        keys = counts.keys()
        for y_i in set(self.y):
            for key_i in keys:
                probability[key_i][y_i] = counts[key_i][y_i]/sum(counts[key][y_i] for key in keys)
        return probability
    def label_pro(self):
        counts = Counter(self.y)
        _counts = defaultdict()
        for y_i in counts.keys():
            _counts[y_i]=counts[y_i]/len(self.y)
        return _counts

    def predict(self,x):
        probability = [1]*len(set(y))
        con_pro= self.conditional_probabilites()

        for feature in x:
            for y_i in set(self.y):
                probability[y_i]*=con_pro[feature][y_i]

        for y_i in set(self.y):
            probability[y_i]*=self.label_pro()[y_i]
        return np.argmax(probability)


if __name__=="__main__":
    X = [['Rainy', 'Hot', 'High', 'False'],
         ['Rainy', 'Hot', 'High', 'True'],
         ['Overcast', 'Hot', 'High', 'False'],
         ['Sunny', 'Mild', 'High', 'False'],
         ['Sunny', 'Cool', 'Normal', 'False'],
         ['Sunny', 'Cool', 'Normal', 'True'],
         ['Overcast', 'Cool', 'Normal', 'True'],
         ['Rainy', 'Mild', 'High', 'False'],
         ['Rainy', 'Cool', 'Normal', 'False'],
         ['Sunny', 'Mild', 'Normal', 'False'],
         ['Rainy', 'Mild', 'Normal', 'True'],
         ['Overcast', 'Mild', 'High', 'True'],
         ['Overcast', 'Hot', 'Normal', 'False'],
         ['Sunny', 'Mild', 'High', 'True']]

    y = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
    model = NaiveBayes(X,y)
    print(model.predict(['Sunny','Mild','High','True']))
