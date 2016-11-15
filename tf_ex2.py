import tensorflow as tf

# Load dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# images 28x28 = 784 pixels flat
x = tf.placeholder(tf.float32, [None, 784]) # None can be any number of inputs. We have 55k images

# Initialize weights and biases
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# model is: W*x + b
y = tf.matmul(x, W) + b
# this is flipped from when we multiplied them in our equation, where we had Wx, as a small trick to deal 
#                                            			  with x being a 2D tensor with multiple inputs.

# initialize correct answers placeholder
y_ = tf.placeholder(tf.float32, [None, 10])

# cross-entropy function = mean (sum (y_ * log(y_predicted)))
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# more numerically stable then the implementation above
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

# can be any kind of optimization. 0.5 is the learning rate
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# model was defined above

# initialize variables for training
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# train the model 1000 times with random 100 points from the training set
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# tf.argmax(y,1) is the label our model thinks is most likely for each input
# tf.argmax(y_,1) is the correct label
  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))  #list of booleans
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  #cast to floating point numbers and then take the mean

# another model defined above for the accuracy computation
  print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))







