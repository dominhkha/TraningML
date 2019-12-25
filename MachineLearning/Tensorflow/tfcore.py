import tensorflow as tf

if __name__ == "__main__":
    t1 = [[1, 2, 3], [4, 5, 6]]
    t2 = [[7, 8, 9], [10, 11, 12]]
    result = tf.concat([t1, t2], axis=1)
    result = tf.expand_dims(t1[0],axis=1)

    result = tf.fill([2,3],10)
    print(result)