import tensorflow as tf

'''
keep_prob = tf.placeholder(tf.float32) # probability to keep units

hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
hidden_layer = tf.nn.relu(hidden_layer)
hidden_layer = tf.nn.dropout(hidden_layer, keep_prob)

logits = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])

'''

'''
tf.nn.dropout()函数有两个参数：

hidden_layer：你要应用 dropout 的 tensor
keep_prob：任何一个给定单元的留存率（没有被丢弃的单元）
keep_prob 可以让你调整丢弃单元的数量。为了补偿被丢弃的单元，tf.nn.dropout() 把所有保留下来的单元（没有被丢弃的单元）* 1/keep_prob

在训练时，一个好的keep_prob初始值是0.5。

在测试时，把 keep_prob 值设为1.0 ，这样保留所有的单元，最大化模型的能力。
'''

# Solution is available in the other "solution.py" tab
import tensorflow as tf

hidden_layer_weights = [
    [0.1, 0.2, 0.4],
    [0.4, 0.6, 0.6],
    [0.5, 0.9, 0.1],
    [0.8, 0.2, 0.8]]
out_weights = [
    [0.1, 0.6],
    [0.2, 0.1],
    [0.7, 0.9]]

# Weights and biases
weights = [
    tf.Variable(hidden_layer_weights),
    tf.Variable(out_weights)]
biases = [
    tf.Variable(tf.zeros(3)),
    tf.Variable(tf.zeros(2))]

# Input
features = tf.Variable([[0.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4], [11.0, 12.0, 13.0, 14.0]])

# TODO: Create Model with Dropout
keep_prob = tf.placeholder(tf.float32)
hidden_layer = tf.add(tf.matmul(features,weights[0]),biases[0])
hidden_layer = tf.nn.relu(hidden_layer)
hidden_layer = tf.nn.dropout(hidden_layer,keep_prob)

logits = tf.add(tf.matmul(hidden_layer,weights[1]),biases[1])

# TODO: Print logits from a session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(logits,feed_dict={keep_prob:0.5}))
