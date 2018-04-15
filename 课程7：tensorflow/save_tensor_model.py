import tensorflow as tf

# 保存文件路径
save_file = './model.ckpt'

# 变量
weights = tf.Variable(tf.truncated_normal([2, 3]))
bias = tf.Variable(tf.truncated_normal([3]))

# 用来存取的Tensor变量的类
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print('Weights:')
    print(sess.run(weights))
    print('Bias:')
    print(sess.run(bias))

    # 保存模型
    saver.save(sess, save_file)


## 加载变量

#移除之前的变量
tf.reset_default_graph()

weights = tf.Variable(tf.truncated_normal([2, 3]))
bias = tf.Variable(tf.truncated_normal([3]))

#用来存取的tf变量
saver = tf.train.Saver()

with tf.Session() as sess:
    #加载权重和权限
    saver.restore(sess,save_file)

    print('Weight:')
    print(sess.run(weights))
    print('Bias:')
    print(sess.run(bias))
