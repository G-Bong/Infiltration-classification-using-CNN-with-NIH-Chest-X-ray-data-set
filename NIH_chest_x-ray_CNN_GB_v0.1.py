import tensorflow as tf
import numpy as np
import random
import cv2
import os
import glob
import pandas as pd
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split

# Bias initialization
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Serve data by batches
def next_batch(batch_size):
    global X_test
    global Y_test
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        X_test = X_test[perm]
        Y_test = Y_test[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return X_test[start:end], Y_test[start:end]

# Convert class labels from scalars to one-hot vectors
# 0 => [1 0] : Non-disease
# 1 => [0 1] : disease
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    # print(num_labels)
    index_offset = np.arange(num_labels) * num_classes
    # print(np.arange(num_labels))
    # print(np.arange(num_labels).shape)
    # print(index_offset)
    labels_one_hot = np.zeros((num_labels, num_classes))
    # print(labels_one_hot)
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


'''
Data Preparation
'''
image_npz_data = np.load('C:/Users/kkb85/Desktop/NIH_Chest_Xray_CNN/x_images_arrays.npz')
image_data = image_npz_data["arr_0"]
print('np.shape(image_data)', np.shape(image_data))
print('image_data.dtype', image_data.dtype)
print('type(image_data)', type(image_data))

print('[0], [1], [2], [3]', np.shape(image_data)[0], np.shape(image_data)[1], np.shape(image_data)[2], np.shape(image_data)[3])



label_npz_data = np.load('C:/Users/kkb85/Desktop/NIH_Chest_Xray_CNN/y_infiltration_labels.npz')
label_data = label_npz_data["arr_0"]
label_count = np.unique(label_data).shape[0]

X_train, X_vali_test, Y_train, Y_vali_test = \
    train_test_split(image_data, label_data, test_size=0.2,
                     random_state=1, stratify=label_data)

X_train = X_train.reshape(-1, 128*128*3)
print('np.shape(X_train)', np.shape(X_train))
Y_train = dense_to_one_hot(Y_train, label_count)
print('Y_train_shape', np.shape(Y_train))




X_vali, X_test, Y_vali, Y_test = \
    train_test_split(X_vali_test, Y_vali_test, test_size=0.5,
                     random_state=1, stratify= Y_vali_test)

X_vali = X_vali.reshape(-1, 128*128*3)
X_test = X_test.reshape(-1, 128*128*3)
print('np.shape(X_vali)', np.shape(X_vali))
print('np.shape(X_test)', np.shape(X_test))
Y_vali = dense_to_one_hot(Y_vali, label_count)
print('Y_vali_shape', np.shape(Y_vali))
Y_test = dense_to_one_hot(Y_test, label_count)
print('Y_test_shape', np.shape(Y_test))

# print('check1', X_vali.shape[0])
# input('stop')

image_size = np.shape(X_train)[1]
print(image_size)


# Parameters
LEARNING_RATE = 0.001
TRAINING_EPOCHS = 500
BATCH_SIZE = 10
DISPLAY_STEP = 10
DROPOUT_CONV = 0.7
DROPOUT_HIDDEN = 0.6
VALIDATION_SIZE = X_vali.shape[0]


'''
Create model with 2D CNN
'''

# Create Input and Output
X = tf.placeholder(tf.float32, shape=[None, image_size], name='input_images')
Y_gt = tf.placeholder(tf.float32, shape=[None, label_count], name='disease_classification')
drop_conv = tf.placeholder('float')
drop_hidden = tf.placeholder('float')

# CNN model
X1 = tf.reshape(X, [-1, 128, 128, 3])

# Layer 1
W1 = tf.get_variable("W1", shape=[2, 2, 3, 32],
                     initializer=tf.contrib.layers.xavier_initializer())
B1 = bias_variable([32])

l1_weight = tf.nn.conv2d(X1, W1, strides=[1, 1, 1, 1], padding='SAME')
l1_relu = tf.nn.relu(l1_weight + B1)
l1_pool = tf.nn.max_pool(l1_relu, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME')
l1_drop = tf.nn.dropout(l1_pool, drop_conv)


# Layer 2
W2 = tf.get_variable("W2", shape=[2, 2, 32, 64],
                     initializer=tf.contrib.layers.xavier_initializer())
B2 = bias_variable([64])

l2_weight = tf.nn.conv2d(l1_drop, W2, strides=[1, 1, 1, 1], padding='SAME')
l2_relu = tf.nn.relu(l2_weight + B2)
l2_pool = tf.nn.max_pool(l2_relu, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME')
l2_drop = tf.nn.dropout(l2_pool, drop_conv)

print('l2_drop_check', np.shape(l2_drop))

# Layer 3 - FC1
W3_FC1 = tf.get_variable("W3_FC1", shape=[32*32*64, 128], initializer=tf.contrib.layers.xavier_initializer())
B3_FC1 = bias_variable([128])
l3_flat = tf.reshape(l2_drop, [-1, W3_FC1.get_shape().as_list()[0]])
l3_feed = tf.nn.relu(tf.matmul(l3_flat, W3_FC1)+ B3_FC1)
l3_drop = tf.nn.dropout(l3_feed, drop_hidden)

# Layer 4 - FC2
W4_FC2 = tf.get_variable("W4_FC2", shape=[128, label_count], initializer=tf.contrib.layers.xavier_initializer())
B4_FC2 = bias_variable([label_count])
Y_pred = tf.nn.softmax(tf.matmul(l3_drop, W4_FC2) + B4_FC2)

# Cost function and training
predict = tf.argmax(Y_pred, 1)
cost = -tf.reduce_sum(Y_gt*tf.log(Y_pred))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
correct_predict = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y_gt, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, 'float'))

'''
TensorFlow Session
'''
epochs_completed = 0
index_in_epoch = 0
num_examples = X_test.shape[0]

# start TensorFlow session
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()

sess.run(init)

# visualisation variables
train_accuracies = []
validation_accuracies = []

DISPLAY_STEP = 1
print("Learning start. It takes sometime.")
for i in range(TRAINING_EPOCHS):

    # get new batch
    batch_xs, batch_ys = next_batch(BATCH_SIZE)

    # check progress on every 1st,2nd,...,10th,20th,...,100th... step
    if i % DISPLAY_STEP == 0 or (i + 1) == TRAINING_EPOCHS:

        train_accuracy = accuracy.eval(feed_dict={X: batch_xs,
                                                  Y_gt: batch_ys,
                                                  drop_conv: DROPOUT_CONV,
                                                  drop_hidden: DROPOUT_HIDDEN})
        if (VALIDATION_SIZE):
            validation_accuracy = accuracy.eval(feed_dict={X: X_vali[0:BATCH_SIZE],
                                                           Y_gt: Y_vali[0:BATCH_SIZE],
                                                           drop_conv: DROPOUT_CONV, drop_hidden: DROPOUT_HIDDEN})
            print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d' % (
            train_accuracy, validation_accuracy, i+1))

            validation_accuracies.append(validation_accuracy)

        else:
            print('training_accuracy => %.4f for step %d' % (train_accuracy, i))
        train_accuracies.append(train_accuracy)

        # increase DISPLAY_STEP
        if i % (DISPLAY_STEP * 10) == 0 and i:
            DISPLAY_STEP *= 10
    # train on batch
    sess.run(optimizer, feed_dict={X: batch_xs, Y_gt: batch_ys, drop_conv: DROPOUT_CONV, drop_hidden: DROPOUT_HIDDEN})
print("Learning Finished!")

# check final accuracy on validation set
if (VALIDATION_SIZE):
    validation_accuracy = accuracy.eval(feed_dict={X: X_vali[0:BATCH_SIZE],
                                                   Y_gt: Y_vali[0:BATCH_SIZE],
                                                   drop_conv: DROPOUT_CONV, drop_hidden: DROPOUT_HIDDEN})
    print('validation_accuracy => %.4f' % validation_accuracy, '\n')



# Test
print('test_images({0[0]},{0[1]})'.format(X_test.shape))


test_accuracy = 0
# predict test set
# using batches is more resource efficient
predicted_lables = np.zeros(X_test.shape[0])
for i in range(0, X_test.shape[0] // BATCH_SIZE):
    predicted_lables[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = predict.eval(
        feed_dict={X: X_test[i * BATCH_SIZE: (i + 1) * BATCH_SIZE], drop_conv: 1.0, drop_hidden: 1.0})

    test_accuracy = accuracy.eval(feed_dict={X: X_test[i * BATCH_SIZE: (i + 1) * BATCH_SIZE], Y_gt: Y_test[i * BATCH_SIZE: (i + 1) * BATCH_SIZE],
                                             drop_conv: 1.0, drop_hidden: 1.0})

    test_accuracy += test_accuracy//(X_test.shape[0] // BATCH_SIZE)

    print('Test accuracy = %.4f' % test_accuracy, '\n')

print(predicted_lables)
print(np.shape(predicted_lables))

# save results
np.savetxt('predict.csv',
           np.c_[range(1, len(X_test) + 1), predicted_lables],
           delimiter=',',
           header='ImageId, Label',
           comments='',
           fmt='%d')

sess.close()