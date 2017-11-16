
# coding: utf-8

# In[1]:


from __future__ import print_function
from skimage.transform import resize

import skimage.io as io 
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Parameter
image_dir = "dog_cat_data/"
image_size = [60,60]
num_trainningset = 20000

# # for test
# image_dir = "test/"
# image_size = [80,80]
# num_trainningset = 600


data_index = []
train_data = []
train_label = []
test_data = []
test_label = []

# Read data info
for dirPath, dirNames, fileNames in os.walk(image_dir):
    print("dataset_path: "+dirPath)
    for f in fileNames:
        data_index.append(os.path.join(dirPath, f))
num_dataset = len(data_index)
num_testingset = num_dataset - num_trainningset

print("This dataset have %s images" %(num_dataset )) 

## =============================== Data preprocessing =============================== ##

for i in range(num_dataset):
    # get random index
    index_rand = int(np.random.rand(1)*len(data_index))
    
    # get this picture name
    pic_name = data_index[index_rand].split('/')[1]
    
    # labeling
    if pic_name.split('.')[0] == 'cat':
        pic_label = [1, 0]
    else:
        pic_label = [0, 1]
    
    # Add the img and lebel to trainning/testing set
    if i < num_trainningset:
        train_data.append(resize( io.imread( data_index[index_rand]), image_size))
        train_label.append(pic_label)
    else:
        test_data.append(resize( io.imread( data_index[index_rand]), image_size))
        test_label.append(pic_label)
    
    data_index.remove(data_index[index_rand])
    
print("Load data finished!!!")
print("Number of trainning data: %s" %(len(train_data)))
print("Number of testing data: %s" %(len(test_data)))

## Plot the distribution of train/test data
# x1 = range(len(train_label[0:50]))
# y1 = train_label[0:50]
# x2 = range(len(test_label[0:50]))
# y2 = test_label[0:50]

# plt.figure(1)
# plt.subplot(211)
# plt.scatter(x1, y1)
# plt.title('Train label')
# plt.xlabel('x')
# plt.ylabel('y')

# plt.subplot(212)
# plt.scatter(x2, y2)
# plt.title('Test label')
# plt.xlabel('x')
# plt.ylabel('y')

# plt.show()

def Next_batch(ind,size):
    out_data = train_data[(ind*size):(ind*size)+size]
    out_index = train_label[(ind*size):(ind*size)+size]
    return out_data, out_index




# In[15]:


## =============================== CNN (3 layer)=============================== ##
# parameter of CNN
lr = 1e-3
training_loop = 2000000
# batch_size = 60 #test
# display_size = 300 #test

batch_size = 500 #total dataset
display_size = 50000 #total dataset

kernel_size = 5 # 5*5
channel_size = 3 # RGB
log_path = "graph/CNN_fullconn"
model_path = "model/CNN_fullconn"
# Number of model
num_input = image_size[0]  # 80*80 per picture 
num_output = 2
num_dropout = 0.8 #Probability of working neuron

# Placeholder
x = tf.placeholder(tf.float32, [None, num_input, num_input, 3], name = 'inputdata')
y = tf.placeholder(tf.float32, [None, num_output], name = 'outputdata')

beta = 0.01
#### Self define function
def AddWeight(shape, name):
#     init_weight = tf.random_normal(shape)
    init_weight = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(init_weight,name = name)

def AddBias(shape, name):
    init_bias = tf.constant(0.1, shape = shape)
    return tf.Variable(init_bias,name = name)

def Conv2D(X, Weight, biase, stride = 1):
    X = tf.nn.conv2d(X, Weight, strides = [1, stride, stride, 1], padding='SAME')
    Y = tf.nn.bias_add(X, biase)
    return(tf.nn.relu(Y))

def MaxPooling(X, stride = 2): # no overlap
    po = tf.nn.max_pool(X, [1, stride, stride, 1], [1, stride, stride, 1], padding='SAME')
    return(po)

def AvgPooling(X, size): # no move
    po = tf.nn.avg_pool(X, [1, size, size, 1], [1, size, size, 1], padding='VALID')
    return(po)

## weights and biases
weights = { 'w_conv1': AddWeight([kernel_size, kernel_size, channel_size, 32], name='w_conv1'),
            'w_conv1_1': AddWeight([1, 1, 32, 32], name='w_conv1_1'),
            'w_conv1_2': AddWeight([1, 1, 32, 32], name='w_conv1_2'),
           
            'w_conv2': AddWeight([kernel_size, kernel_size, 32, 64], name='w_conv2'),
            'w_conv2_1': AddWeight([1, 1, 64, 64], name='w_conv2_1'),
            'w_conv2_2': AddWeight([1, 1, 64, 64], name='w_conv2_2'),
           
            'w_conv3': AddWeight([kernel_size, kernel_size, 64, 128], name='w_conv3'),
            'w_conv3_1': AddWeight([1, 1, 128, 128], name='w_conv3_1'),
            'w_conv3_2': AddWeight([1, 1, 128, 128], name='w_conv3_2'),
           
            'w_fullconn': AddWeight([1* 1* 128, 512], name='w_fullconn'),
            'w_out': AddWeight([512, num_output], name='w_out')}

biases = {'b_conv1': AddBias([32], name = 'b_conv1'), 
          'b_conv1_1': AddBias([32], name = 'b_conv1_1'),
          'b_conv1_2': AddBias([32], name = 'b_conv1_2'),
          
          'b_conv2': AddBias([64], name = 'b_conv2'),
          'b_conv2_1': AddBias([64], name = 'b_conv2_1'),
          'b_conv2_2': AddBias([64], name = 'b_conv2_2'),
          
          'b_conv3': AddBias([128], name = 'b_conv3'),
          'b_conv3_1': AddBias([128], name = 'b_conv3_1'),
          'b_conv3_2': AddBias([128], name = 'b_conv3_2'),

          'b_fullconn': AddBias([512], name = 'b_fullconn'), 
          'b_out': AddBias([num_output], name = 'b_out')}

## CNN model
def CNN(X, weights, biases, isTrain):
    ## input to cell
    ## X(20 batch, x:80, y:80, channel:3) => (20,80,80,3)
    X = tf.reshape(X, [-1, image_size[0], image_size[1], channel_size])

    ## Converlution layer1 
    X_in = Conv2D(X, weights['w_conv1'], biases['b_conv1'])
    X_in11 = Conv2D(X_in, weights['w_conv1_1'], biases['b_conv1_1'])
    X_in12 = Conv2D(X_in11, weights['w_conv1_2'], biases['b_conv1_2'])
    # Pooling layer1
    h_pool1 = MaxPooling(X_in12)
    ImgSizePool = num_input/2;
    
    ## Converlution layer2 
    X_in2 = Conv2D(h_pool1, weights['w_conv2'], biases['b_conv2'])
    X_in21 = Conv2D(X_in2, weights['w_conv2_1'], biases['b_conv2_1'])
    X_in22 = Conv2D(X_in21, weights['w_conv2_2'], biases['b_conv2_2'])
    # Pooling layer2
    h_pool2 = MaxPooling(X_in22)
    ImgSizePool = ImgSizePool/2
    
    ## Converlution layer3
    X_in3 = Conv2D(h_pool2, weights['w_conv3'], biases['b_conv3'])
    X_in31 = Conv2D(X_in3, weights['w_conv3_1'], biases['b_conv3_1'])
    X_in32 = Conv2D(X_in31, weights['w_conv3_2'], biases['b_conv3_2'])
    # Pooling layer3
    h_pool3 = AvgPooling(X_in32, ImgSizePool)
    
    ## Fully connection    
    h_pool3_reshape = tf.reshape(h_pool3, [-1, weights['w_fullconn'].get_shape().as_list()[0]])
    h_fullconn = tf.nn.relu(tf.add(tf.matmul(h_pool3_reshape, weights['w_fullconn']), biases['b_fullconn']))
    
    if(isTrain == True):
        ## Dropout
#         h_drop = tf.nn.dropout(h_fullconn, num_dropout)
        h_drop = h_fullconn
    else:
        h_drop = h_fullconn
    
    return tf.nn.softmax(tf.add(tf.matmul(h_drop, weights['w_out']),biases['b_out']))

with tf.name_scope('Model'):
    predict = CNN(x, weights, biases, True)
    
with tf.name_scope('test_Model'):
    predict_test = CNN(x, weights, biases, False)

with tf.name_scope('Loss'):
#     predict_tmp = tf.clip_by_value(predict,1e-8,1.0)
    predict_tmp = predict
    # L2 regularization
    loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(predict_tmp), reduction_indices = 1))
#     regularizers = tf.nn.l2_loss(weights['w_conv1']) + tf.nn.l2_loss(weights['w_conv1_1']) +tf.nn.l2_loss(weights['w_conv1_2']) + \
#                    tf.nn.l2_loss(weights['w_conv2']) + tf.nn.l2_loss(weights['w_conv2_1']) +tf.nn.l2_loss(weights['w_conv2_2']) + \
#                    tf.nn.l2_loss(weights['w_conv3']) + tf.nn.l2_loss(weights['w_conv3_1']) +tf.nn.l2_loss(weights['w_conv3_2']) + \
#                    tf.nn.l2_loss(weights['w_fullconn']) + \
#                    tf.nn.l2_loss(weights['w_out'])
#     loss = tf.reduce_mean(loss + beta * regularizers)
#     loss = predict
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y))

with tf.name_scope('Update'):
    ## update weights and biases
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

with tf.name_scope('Accuracy'):
    ## evaluate
    correcr_predict = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1)) # 0 is col; 1 is row
    accuracy = tf.reduce_mean(tf.cast(correcr_predict, tf.float32))
    
with tf.name_scope('test_Accuracy'):
    ## evaluate
    correcr_predict_test = tf.equal(tf.argmax(predict_test, 1), tf.argmax(y, 1)) # 0 is col; 1 is row
    accuracy_test = tf.reduce_mean(tf.cast(correcr_predict_test, tf.float32))

def test_data_proc(step_now, data_size):

    test_d = test_data[step_now*data_size:(step_now*data_size) + data_size]
    test_ind = test_label[step_now*data_size:(step_now*data_size) + data_size]
    return test_d, test_ind    

## Save for graph
tf.summary.scalar("Loss", loss)
tf.summary.scalar("Accuracy", accuracy)
sum_op = tf.summary.merge_all()

saver = tf.train.Saver()

#==================================================================#

## Run sess
sess = tf.Session() 
sess.run(tf.global_variables_initializer())

if tf.gfile.Exists(log_path):
    tf.gfile.DeleteRecursively(log_path)
tf.gfile.MakeDirs(log_path)

summary_writer = tf.summary.FileWriter(log_path, sess.graph)

step = 0
Num_slice_test = 10
data_size_step = len(test_data)/Num_slice_test
testing_acc_old = 0

while step*batch_size < training_loop:
    batch_x, batch_y = Next_batch((step%10), batch_size)
#     batch_x = batch_x.reshape([batch_size, num_input, 3])
    ## *******feed_dict cannot feed Tensor, so we use batch_x.reshape******
    # batch_x = tf.reshape(batch_x, [batch_size, num_timestamp, num_input])
#     batch_x = batch_x.reshape([batch_size, num_timestamp, num_input])
#     print(len(batch_y))
    sess.run(optimizer, feed_dict = {x: batch_x, y: batch_y})

    if step*batch_size % display_size == 0:
        acc = sess.run( accuracy, feed_dict = {x:batch_x, y:batch_y})
        los = sess.run( loss, feed_dict = {x:batch_x, y:batch_y})
#         if(tf.check_numerics(los) == InvalidArgument)
#             print("NAN!!!")
        
        print("Iterate: %s" %(step))
        print("============ Train step ============")
        print("Loss: %s" %(los))
        print("Accuracy of trainning: %s" %(acc))

        ## ===================== Testing ===================== ## 
        print("============ Test step ============")
        print("test data total size: %s" %(len(test_label)))
        step_test = 0
        acc_all = 0

        while (step_test<Num_slice_test):
            in_test, label_test = test_data_proc(step_test,data_size_step)
            acc_test = sess.run( accuracy_test, feed_dict = {x:in_test, y:label_test})
#             print("Testdata%s~%s " %(step_test*data_size_step, step_test*data_size_step + data_size_step - 1))
#             print("accuracy: %s" %(acc_test))
            acc_all+=acc_test
            step_test+=1
        
        testing_acc = round(acc_all/Num_slice_test, 5)
        print("Average accuracy of testing: %s" %(testing_acc))
        
        if(testing_acc > testing_acc_old):
            # Save the model
            if tf.gfile.Exists(model_path):
                tf.gfile.DeleteRecursively(model_path)
            save_path = saver.save(sess, model_path)
            print("Model saved in %s" % save_path)
            testing_acc_old = testing_acc
    step+=1

## Save all infomation 
# save_path = saver.save(sess, model_path)
# print("Model saved in %s" % save_path)

sess.close()
print("END!!!!!!!!!!!!!!!!!!!!!")
print("The Best testing accuracy: %s" %(testing_acc_old))
## if want to load saved model
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

# ## Load model
# saver.restore(sess, model_path)

