import numpy as np
import re
from collections import defaultdict,Counter


class NaiveBayes:

    def counts(self,X,y):
        counts = defaultdict(lambda : [1,1])
        for index,x_i in enumerate(X):
            for x_i_j in x_i:
                counts[x_i_j][y[index]]+=1
        return counts
    
    def conditional_probabilites(self,X,y):
        counts = self.counts(X,y)
        keys = counts.keys()
        probability = defaultdict(lambda : [0,0])
        for key_i in keys:
            probability[key_i][0] = counts[key_i][0]/(sum(counts[key][0] for key in counts.keys()))
            probability[key_i][1] = counts[key_i][1]/(sum(counts[key][1] for key in counts.keys()))
        return probability

    def yes_no_probability(self,y):
        counts = Counter(y)
        _counts = defaultdict()
        for key in counts.keys():
            _counts[key]=counts[key]/(sum(counts.values()))
        return _counts
    
    def predict(self,X,y,X_test):
        p_yes = 1
        p_no = 1
        con_pro = self.conditional_probabilites(X,y)
        for feature in X_test:
            p_yes *= con_pro[feature][1]
            p_no *= con_pro[feature][0]
        p_yes*=self.yes_no_probability(y)[1]
        p_no*=self.yes_no_probability(y)[0]
        return "yes" if p_yes> p_no else "no"
    

if __name__ == "__main__":
    
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
    
    y = [0,0,1,1,1,0,1,0,1,1,1,1,1,0]

    model = NaiveBayes()
    print(model.predict(X,y,['Sunny','Mild','High','True']))

