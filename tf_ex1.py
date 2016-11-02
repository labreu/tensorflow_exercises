import pandas as pd
import tensorflow as tf

a = tf.constant(10)
b = tf.constant(12)

x = tf.add(a, b)
sess = tf.Session()

print sess.run(x)
