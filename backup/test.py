import tensorflow as tf

inputs = tf.constant([[1,2,3,4,5],[2,4,6,8,10],[3,6,9,12,15],[4,8,12,16,20],[5,10,15,20,25]], tf.float32)
input_tensor = tf.reshape(inputs, [1,5,5,1])

#new_w = tf.constant([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]], tf.float32)
#new_w = tf.reshape(new_w, [4,4,1,1])
new_w = tf.constant([[1,1,-1],[1,1,-1],[1,1,-1]], tf.float32)
new_w = tf.reshape(new_w, [3,3,1,1])

conv1_biases = tf.get_variable("bias", 1, initializer=tf.constant_initializer(0.0))
conv1_weights = tf.get_variable(
            "weight",
            #[4, 4, 1, 1],
            [3, 3, 1, 1],
            initializer=tf.constant_initializer(1.0))
conv1_weights1 = tf.assign(conv1_weights, new_w)

conv1 = tf.nn.conv2d(input_tensor, conv1_weights1, strides=[1, 1, 1, 1], padding='SAME')
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
pool1 = tf.nn.avg_pool(input_tensor, ksize=[1,3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
#pool1 = tf.nn.avg_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    output = sess.run(pool1)
    #print(output.reshape([5,5]))
    print(output.reshape([1,-1]))

