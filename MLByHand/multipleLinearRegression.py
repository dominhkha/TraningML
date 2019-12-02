
# Multiple Linear Regression
# dominhkha
#2/12/19

import numpy as np 

class MultipleLinearRegression:

    def target_fn(self,x,theta):
        return sum(x_i*theta_i for x_i,theta_i in zip(x,theta))

    def gradient_sum_square(self,x,y,theta,x_j):
        return sum((y_i - self.target_fn(x_i, theta)) for x_i,y_i in zip(x,y))
    
    def update(self,x,y,theta,learning_rate=0.001):
        # return [theta_i - learning_rate * self.gradient_sum_square(x,y,theta,x_i_j)  for theta_i,x_i_j in zip(theta,x_i) for x_i in x]
        new_theta = []
        for index,theta_i in enumerate(theta):
            theta_i = theta_i-learning_rate * 0.2* sum([(self.target_fn(x_i,theta)-y_i)*x_i[index] for x_i,y_i in zip(x,y)])
            new_theta.append(theta_i)
        return new_theta
    
    def distance_between_two_vectors(self,v1,v2):
        return sum((v1_i-v2_i)**2 for v1_i,v2_i in zip(v1,v2))

    def sum_square(self,x,y,theta):
        return sum((self.target_fn(x_i,theta)-y_i)**2 for x_i,y_i in zip(x,y))
    def batchGradientDescend(self,x,y,theta):

        while True:
            next_theta = self.update(x,y,theta)
            print(next_theta)
            print(self.sum_square(x,y,next_theta))
            if self.distance_between_two_vectors(next_theta,theta) < 0.0000001:
                break
            theta = next_theta

if __name__ =="__main__":
    real_theta = [1,2,3]
    x = [[1,2,3],[4,5,6],[2,3,4],[3,6,7],[6,1,2]]
    y = [sum(x_i_j*theta_j for x_i_j,theta_j in zip(x_i,real_theta) ) for x_i in x]
    model = MultipleLinearRegression()
    model.batchGradientDescend(x,y,[0.5,1.5,2.5])

