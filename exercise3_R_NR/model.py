import tensorflow as tf

class Model:
    
    def __init__(self,  input_shape,  n_classes, lr):
        
        history_length = input_shape[2]
        # TODO: Define network
        # ...
        self.X = tf.placeholder("float",  [None,  input_shape[0],  input_shape[1], history_length])
        self.y = tf.placeholder("float", [None,  n_classes] )
        
        # conv layers
        
        conv_layer1 = tf.layers.conv2d(self.X,  16,  3,  padding="same",  activation=tf.nn.relu)
        pool_layer1 = tf.layers.max_pooling2d(conv_layer1,  2,  1,  padding = "same")
        conv_layer2 = tf.layers.conv2d(pool_layer1,  16,  3,  padding="same",  activation=tf.nn.relu)
        pool_layer2 = tf.layers.max_pooling2d(conv_layer2,  2,  1,  padding = "same")
        
        # dense_layers
        flat_layer1 = tf.layers.flatten(pool_layer2)
        dense_layer1 = tf.layers.dense(flat_layer1,  128,  activation=tf.nn.relu)
        dense_layer2 = tf.layers.dense(dense_layer1,  n_classes)
        self.output = tf.nn.softmax(dense_layer2)
        
        # TODO: Loss and optimizer
        
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=self.y))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)
        
        
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.output, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        # TODO: Start tensorflow session
        
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def load(self, file_name):
        self.saver.restore(self.sess, file_name)

    def save(self, file_name):
        self.saver.save(self.sess, file_name)
        
