import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class LinearRegrerssion:
    def __init__(self):
        self.W = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x):
        return self.W * x + self.b

    def loss(self, predicted_y, true_y):
        return tf.reduce_mean(tf.square(predicted_y - true_y))

    def train(self, model, inputs, true_y, learning_rate):
        with tf.GradientTape() as t:
            loss = self.loss(model(inputs), true_y)
        dW, db = t.gradient(loss, [self.W, self.b])
        model.W.assign_sub(learning_rate*dW)
        model.b.assign_sub(learning_rate*db)


if __name__ == "__main__":
    true_W = 3.0
    true_b = 2.0
    num_sample = 1000
    epochs = 100

    inputs = tf.random.normal(shape=[num_sample])

    true_y = inputs * true_W + true_b
    model = LinearRegrerssion()

    Ws, bs = [], []
    loss = []
    for epoch in range(epochs):
        Ws.append(model.W.numpy())
        bs.append(model.b.numpy())
        current_loss= model.loss(model(inputs), true_y)
        loss.append(current_loss)
        model.train(model, inputs, true_y, learning_rate=0.2)
        print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' %
              (epoch, Ws[-1], bs[-1], current_loss))

    plt.plot(range(epochs),loss)
    plt.show()

