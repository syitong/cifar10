import tensorflow as tf
import numpy as np

# to download CIFAR-10 data with helper module "load_data"
from keras.datasets.cifar10 import load_data

# define next_batch function to read next batch
def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

# construc CNN model
def build_CNN_classifier(x):
  """provide CNN graph to classify CIFAR-10 images
  Args:
    x: (N_examples, 32, 32, 3) dimensioanl input tensor
  Returns:
    tuple (y, keep_prob).
    y is tensor digit(0-9) in the form of (N_examples, 10)
    keep_prob is scalar placeholder for dropout
  """
  # input image
  x_image = x

  # 1st convolutional layer - mapping from a grayscale image to 64 features
  W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 64], stddev=5e-2))
  b_conv1 = tf.Variable(tf.constant(0.1, shape=[64]))
  l_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
  mean,variance = tf.nn.moments(l_conv1,axes=[0],keep_dims=True)
  gamma1 = tf.Variable(tf.truncated_normal(shape=tf.shape(variance),stddev=5e-2))
  beta1 = tf.Variable(tf.truncated_normal(shape=tf.shape(mean),stddev=5e-2))
  n_conv1 = tf.nn.batch_normalization(l_conv1,mean,tf.sqrt(variance),beta1,gamma1,0.1)
  h_conv1 = tf.nn.relu(n_conv1)

  # 1st Pooling layer
  h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

  # 2nd convolutional layer - mapping from 32 features to 64 features
  W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 64, 64], stddev=5e-2))
  b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
  h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

  # 2nd pooling layer.
  h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

  # 3rd convolutional layer
  W_conv3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], stddev=5e-2))
  b_conv3 = tf.Variable(tf.constant(0.1, shape=[128]))
  h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)

  # 4th convolutional layer
  W_conv4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2))
  b_conv4 = tf.Variable(tf.constant(0.1, shape=[128]))
  h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4)

  # 5th convolutional layer
  W_conv5 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2))
  b_conv5 = tf.Variable(tf.constant(0.1, shape=[128]))
  h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, W_conv5, strides=[1, 1, 1, 1], padding='SAME') + b_conv5)

  # Fully Connected Layer 1 -- after downsampling twice, 32x32 image be 8x8x128 feature map
  # And then mapping to 384 features
  W_fc1 = tf.Variable(tf.truncated_normal(shape=[8 * 8 * 128, 384], stddev=5e-2))
  b_fc1 = tf.Variable(tf.constant(0.1, shape=[384]))

  h_conv5_flat = tf.reshape(h_conv5, [-1, 8*8*128])
  h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)

  # Dropout - control complexity, prevent feature's co-adaptation
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # mapping from 384 features to 10 classes-airplane, automobile, bird...
  W_fc2 = tf.Variable(tf.truncated_normal(shape=[384, 10], stddev=5e-2))
  b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
  logits = tf.matmul(h_fc1_drop,W_fc2) + b_fc2
  y_pred = tf.nn.softmax(logits)

  return y_pred, logits

# define placeholder for input and output, probability for dropout
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

# download and load CIFAR-10
(x_train, y_train), (x_test, y_test) = load_data()
# transform scalar label into One-hot Encoding
y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10),axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10),axis=1)

# generate graph of Convolutional Neural Networks(CNN)
y_pred, logits = build_CNN_classifier(x)

# define Cross Entropy as loss function, minimize it with RMSPropOptimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_step = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

# compute accuracy
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# run session
with tf.Session() as sess:
  # initialize
  sess.run(tf.global_variables_initializer())

  # optimize 1000 Step
  for i in range(10000):
    batch = next_batch(128, x_train, y_train_one_hot.eval())

    # print accuracy and loss every 100 Step
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
      loss_print = loss.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})

      print("Epoch: %d, training accuracy: %f, loss: %f" % (i, train_accuracy, loss_print))
    # use Dropout with 20% probability
    sess.run(train_step, feed_dict={x: batch[0], y: batch[1], keep_prob: 0.8})

  # print test accuracy after completing training
  test_accuracy = 0
  for i in range(100):
      test_batch = next_batch(100, x_test, y_test_one_hot.eval())
      test_accuracy += accuracy.eval(feed_dict={x: test_batch[0], y: test_batch[1], keep_prob: 1.0})
  test_accuracy = test_accuracy / 100
  print("Test accuracy: %f" % test_accuracy)
