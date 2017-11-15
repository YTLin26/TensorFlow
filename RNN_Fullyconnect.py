from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

## Load data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

## Parameter
lr = 0.001
training_loop = 10000
batch_size = 100
display_size = 5000
log_path = "graph/RNN_fullconn"
model_path = "model/RNN_fullconn"

## number of model
num_input = 28  #28 per oneline
num_timestamp = 28 #28 
num_hidden = 128
num_output = 10

## Placeholder
x = tf.placeholder(tf.float32, [None, num_timestamp ,num_input], name = 'inputdata')
y = tf.placeholder(tf.float32, [None, num_output], name = 'outputdata')

#### Self define function
def AddWeight(shape, name):
    init_weight = tf.random_normal(shape)
    return tf.Variable(init_weight,name = name)

def AddBias(shape, name):
    init_bias = tf.constant(0.1, shape = shape)
    return tf.Variable(init_bias,name = name)

## weights and biases
weights = {'w_hidden': AddWeight([num_input,num_hidden], name='Weight_Hidden'),
          'w_out': AddWeight([num_hidden,num_output], name='Weight_out')}

biases = {'b_hidden': AddBias([num_hidden], name = 'Bias_hidden'), 
        'b_out': AddBias([num_output], name = 'Bias_out')}

## RNN static model
def RNN2(X, weights, biases):
    # unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    X = tf.unstack(X, num_timestamp, 1)

    # define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias = 1.0)
    # get lstm cell output
    outputs, states = tf.nn.static_rnn(lstm_cell, X, dtype = tf.float32)

    # linear activation, using rnn inner loop last output
    return tf.nn.softmax(tf.add(tf.matmul(outputs[-1], weights['w_out']),biases['b_out']))

## RNN dynamic model
def RNN(X, weights, biases):
    ## input to cell
    ## X(100 batch, 28 step, 28 input) => (100*28,28)
    print(X.shape)
    X = tf.reshape(X, [-1, num_input])
    X_in = tf.matmul(X, weights['w_hidden']) + biases['b_hidden']

    ## (batch*step,input) => (batch, step, input)
    X_in = tf.reshape(X_in, [-1, num_timestamp, num_hidden] )
    
    ## Setting forget gate, ...
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=1.0)#, state_is_tuple=False)

    ## RNN dynamic model
    init_state = cell.zero_state(batch_size,dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
    outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))

    ## return output layer
    return  tf.nn.softmax(tf.add(tf.matmul(outputs[-1], weights['w_out']),biases['b_out']))

with tf.name_scope('Model'):
    predict = RNN(x, weights, biases)

with tf.name_scope('Loss'):
    # loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(predict), reduction_indices = 1))
    loss = predict
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y))

with tf.name_scope('Update'):
    ## update weights and biases
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

with tf.name_scope('Accuracy'):
    ## evaluate
    correcr_predict = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1)) # 0 is col; 1 is row
    accuracy = tf.reduce_mean(tf.cast(correcr_predict, tf.float32))

tf.summary.scalar("Loss", loss)
tf.summary.scalar("Accuracy", accuracy)
sum_op = tf.summary.merge_all()

saver = tf.train.Saver()

#==================================================================#

## Run sess
sess = tf.Session() 
sess.run(tf.global_variables_initializer())

# summary_writer = tf.summary.FileWriter(log_path, graph = tf.get_default_graph())


if tf.gfile.Exists(log_path):
    tf.gfile.DeleteRecursively(log_path)
tf.gfile.MakeDirs(log_path)
summary_writer = tf.summary.FileWriter(log_path, sess.graph)

step = 0
while step*batch_size < training_loop:
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    # print( batch_y[0])
    ## *******feed_dict cannot feed Tensor, so we use batch_x.reshape******
    # batch_x = tf.reshape(batch_x, [batch_size, num_timestamp, num_input])
    batch_x = batch_x.reshape([batch_size, num_timestamp, num_input])

    sess.run(optimizer, feed_dict = {x: batch_x, y: batch_y})

    if step*batch_size % display_size == 0:
        acc = sess.run( accuracy, feed_dict = {x:batch_x, y:batch_y})
        los = sess.run( loss, feed_dict = {x:batch_x, y:batch_y})
        print("Iterate: %s" %(step))
        print("Loss: %s" %(los))
        print("accuracy: %s" %(acc))
    step+=1

## Save all infomation 
save_path = saver.save(sess, model_path)
print("Model saved in %s" % save_path)

sess.close()
print("END!!!!!!!!!!!!!!!!!!!!!")

## if want to load saved model
sess = tf.Session()
sess.run(tf.global_variables_initializer())

## Load model
saver.restore(sess, model_path)

## run test image
# test_x = mnist.test.images
# test_y = mnist.test.labels
# test_x = test_x.reshape([10000, num_timestamp, num_input])
# print(test_x.shape)
# print(test_y.shape)
# acc_test = sess.run(accuracy, feed_dict = {x: test_x, y: test_y})
# los_test = sess.run(loss, feed_dict = {x:test_x, y: test_y})

# print("Testing image accuracy: %s" %(acc_test))
# print("Testing image loss: %s" %(los_test))
print("If want to see graph on tensorboard: tensorboard --logdir=%s" %(log_path))


