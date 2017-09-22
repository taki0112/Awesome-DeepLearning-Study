
import tensorflow as tf
import cifar10
from ops import *

class TripleGAN(object) :
    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name, checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.model_name = "TripleGAN"     # name for checkpoint

        if self.dataset_name == 'cifar10' :
            self.input_height = 32
            self.input_width = 32
            self.output_height = 32
            self.output_width = 32

            self.z_dim = z_dim
            self.c_dim = 3

            self.learning_rate = 3e-4
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8

            self.sample_num = 64

            self.data_X, self.data_y, _, _, = cifar10.prepare_data() # trainX, trainY, testX, testY

            self.num_batches = len(self.data_X) // self.batch_size

        else :
            raise NotImplementedError

    def discriminator(self, x, c, scope='discriminator', is_training=True, reuse=False):
        with tf.variable_scope(scope, reuse=reuse) :
            x = dropout(x, rate=0.2, is_training=is_training)
            x = concat([z, c])

            x = lrelu(conv_layer(x, filter=32, kernel=[3,3], layer_name=scope+'_conv1'))
            x = concat([x, c])
            x = lrelu(conv_layer(x, filter=32, kernel=[3,3], stride=2, layer_name=scope+'_conv2'))
            x = dropout(x, rate=0.2, is_training=is_training)
            x = concat([x, c])

            x = lrelu(conv_layer(x, filter=64, kernel=[3,3], layer_name=scope+'_conv3'))
            x = concat([x, c])
            x = lrelu(conv_layer(x, filter=64, kernel=[3,3], stride=2, layer_name=scope+'_conv4'))
            x = dropout(x, rate=0.2, is_training=is_training)
            x = concat([x, c])

            x = lrelu(conv_layer(x, filter=128, kernel=[3,3], layer_name=scope+'_conv5'))
            x = concat([x, c])
            x = lrelu(conv_layer(x, filter=128, kernel=[3,3], layer_name=scope+'_conv6'))
            x = concat([x, c])

            x = Global_Average_Pooling(x)
            x = flatten(x)
            x = concat([x,c]) # 맞나 ?

            x = linear(x, output=1, layer_name=scope+'_linear1')
            x = sigmoid(x)

            return x

    def generator(self, z, c, scope='generator', is_training=True, reuse=False):
        with tf.variable_scope(scope, reuse=reuse) :

            x = concat([z,c]) # 맞나 ?

            x = relu(linear(x, output=512*4*4, layer_name=scope+'_linear1'))
            x = batch_norm(x, is_training=is_training, scope=scope+'_batch1')

            x = tf.reshape(x, shape=[-1, 4, 4, 512]) # -1이 아니라 self.batch_size인가 ?
            x = concat([x, c])

            x = relu(deconv_layer(x, filter=256, kernel=[5,5], stride=2, layer_name=scope+'_deconv1'))
            x = batch_norm(x, is_training=is_training, scope=scope+'_batch2')
            x = concat([x, c])

            x = relu(deconv_layer(x, filter=128, kernel=[5,5], stride=2, layer_name=scope+'_deconv2'))
            x = batch_norm(x, is_training=is_training, scope=scope+'_batch3')
            x = concat([x, c])

            x = tanh(deconv_layer(x, filter=3, kernel=[5,5], stride=2, layer_name=scope+'deconv3'))

            return x

    def classifier(self, x, scope='classifier', is_training=True, reuse=True):
        with tf.variable_scope(scope, reuse=reuse) :
            x = lrelu(conv_layer(x, filter=128, kernel=[3,3], layer_name=scope+'_conv1'))
            x = lrelu(conv_layer(x, filter=128, kernel=[3,3], layer_name=scope + '_conv2'))
            x = lrelu(conv_layer(x, filter=128, kernel=[3,3], layer_name=scope + '_conv3'))

            x = max_pooling(x, kernel=[2,2], stride=2)
            x = dropout(x, rate=0.5, is_training=is_training)

            x = lrelu(conv_layer(x, filter=256, kernel=[3,3], layer_name=scope+'_conv4'))
            x = lrelu(conv_layer(x, filter=256, kernel=[3,3], layer_name=scope+'_conv5'))
            x = lrelu(conv_layer(x, filter=256, kernel=[3,3], layer_name=scope+'_conv6'))

            x = max_pooling(x, kernel=[2,2], stride=2)
            x = dropout(x, rate=0.5, is_training=is_training)

            x = lrelu(conv_layer(x, filter=512, kernel=[3,3], layer_name=scope+'_conv7'))

            """
            x = lrelu(conv_layer(x, filter=256, kernel=[3,3], layer_name=scope+'_conv8'))
            x = lrelu(conv_layer(x, filter=128, kernel=[3,3], layer_name=scope+'_conv9'))
            """


            x = Global_Average_Pooling(x)
            x = flatten(x)

            x = linear(x, output=10, layer_name=scope+'_linear1')

            return x

