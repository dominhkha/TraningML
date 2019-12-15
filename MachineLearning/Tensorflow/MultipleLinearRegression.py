import numpy as np
import tensorflow as tf


class MulipleLinearRegression:
    def __init__(self):
        self.W = tf.Variable([1.0, 1.0, 1.0])

    def __call__(self, x):
        return self.W * x

    def loss(self, predicted, true):
        return tf.reduce_mean(tf.square(true - predicted))

    def train(self, model, x, y, learning_rate):
        with tf.GradientTape() as t:
            loss = self.loss(model(x), y)
        dW = t.gradient(loss, self.W)
        self.W.assign_sub(learning_rate * dW)


if __name__ == "__main__":
    num_sample = 1000
    epochs = 100
    true_W = [1, 2, 3]
    x = tf.random.normal(shape=[num_sample, 2])
    x = tf.concat([x, tf.ones(shape=[num_sample, 1])], axis=1)
    y = x * true_W
    model = MulipleLinearRegression()
    Ws = []
    loss = []
    for epoch in range(epochs):
        Ws.append(model.W.numpy())
        current_loss = model.loss(model(x), y)
        loss.append(current_loss)
        model.train(model, x, y, learning_rate=0.02)
        print("epoch %2d: W1=%1.2f W2=%1.2f W3=%1.2f, loss=%2.5f" % (epoch, Ws[-1][0], Ws[-1][1], Ws[-1][2], loss[-1]))
