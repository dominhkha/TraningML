
# dominhkha
# 3/12/19

import numpy as np 
import math

class Logistic :

    def logistic(sefl,x ):
        return 1.0/(1+math.exp(-x))
    
    def logistic_prime(self,x):
        return self.logistic(x) * (1- self.logistic(x))
    
    def logistic_log_likelihood_i(self,x_i,y_i,beta):
        if y_i ==1:
            return math.log(self.logistic(np.dot(x_i,beta)))
        else :
            return math.log(1-self.logistic(np.dot(x_i,beta)))
    
    def logistic_log_likelihoodl(self,x,y,beta):
        return sum(self.logistic_log_likelihood_i(x_i,y_i,beta) for x_i,y_i in zip(x,y))


