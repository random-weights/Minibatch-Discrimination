import tensorflow as tf
import numpy as np
from Helper import *


x = tf.placeholder(tf.float32,shape = [None,28,28,1])
y_true = tf.placeholder(tf.float32,shape = [None,10])

y_true_cls = tf.argmax(y_true,axis = 1)

ls_filt_count = [16,32,64]

kern_init = tf.random_normal_initializer(mean = 0.0, stddev = 0.1)
bias_init = tf.zeros_initializer()

layer = x
for i in range(len(ls_filt_count)):
    layer = tf.layers.conv2d(layer,ls_filt_count[i],[5,5],[1,1],'same',kernel_initializer=kern_init,bias_initializer=bias_init)
    layer = tf.layers.max_pooling2d(layer,[3,3],[1,1],'same')
    layer = tf.contrib.layers.batch_norm(layer,scope = "layer"+str(i)+"bn")
    layer = tf.nn.relu(layer,name = "layer"+str(i))

flat_tensor = tf.layers.flatten(layer,name = "flat_tensor")

ls_units_count = [256, 64, 10]
a = flat_tensor
for i in range(len(ls_units_count)):
    z = tf.layers.dense(a,ls_units_count[i],kernel_initializer=kern_init,bias_initializer=bias_init)
    z = tf.contrib.layers.batch_norm(z,scope = "layer"+str(i+3)+"bn")
    a = tf.nn.relu(z,name = "layer"+str(i+3))

y_pred = tf.nn.softmax(z)
y_pred_cls = tf.argmax(y_pred,axis = 1,name = "output_class")

correct_pred = tf.equal(y_pred_cls,y_true_cls,name = "correct_pred")
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32),name = 'accuracy')*100

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = z,labels = y_true))
train = tf.train.AdamOptimizer(1e-2).minimize(loss)

epochs = 1000
batch_size = 32

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    train_data = Data()
    train_data.get_xdata("data/x_train.csv")
    train_data.get_ydata("data/y_train.csv")

    for i in range(epochs):
        train_data.get_rand_batch(batch_size)
        x_batch = train_data.x_batch
        y_batch = train_data.y_batch
        feed_dict = {x:x_batch,y_true: y_batch}

        _,cost,accu = sess.run([train,loss,accuracy],feed_dict)
        print("Epoch: ",str(i+1)+"\tcost: ",cost,"\tAcc_train: ",accu)