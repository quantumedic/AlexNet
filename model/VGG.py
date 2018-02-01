import tensorflow as tf
import math
from .. import layer

def alexNet(x, w, h):
	keep_prob = tf.placeholder(tf.float32)

	with tf.name_scope('reshape'):
		net = tf.reshaped(x, [-1, w, h, 3])

	with tf.name_scope('conv1'):
		net = layer.conv(net, input_chanel=3, num_output=64, kernel_size=3, stride=1, 'VALID')
		net = layer.conv(net, input_chanel=64, num_output=64, kernel_size=3, stride=1, 'VALID')

	with tf.name_scope('pool1'):
		net = layer.max_pooling(net)

	with tf.name_scope('conv2'):
		net = layer.conv(net, input_chanel=64, num_output=128, kernel_size=3, stride=1, 'SAME')
		net = layer.conv(net, input_chanel=128, num_output=128, kernel_size=3, stride=1, 'SAME')
		
	with tf.name_scope('pool2'):
		net = layer.max_pooling(net)

	with tf.name_scope('conv3'):
		net = layer.conv(net, input_chanel=128, num_output=256, kernel_size=3, stride=1, 'SAME')
		net = layer.conv(net, input_chanel=256, num_output=256, kernel_size=3, stride=1, 'SAME')
		net = layer.conv(net, input_chanel=256, num_output=256, kernel_size=3, stride=1, 'SAME')

	with tf.name_scope('pool3'):
		net = layer.max_pooling(net)

	with tf.name_scope('conv4'):
		net = layer.conv(net, input_chanel=256, num_output=512, kernel_size=3, stride=1, 'SAME')
		net = layer.conv(net, input_chanel=512, num_output=512, kernel_size=3, stride=1, 'SAME')
		net = layer.conv(net, input_chanel=512, num_output=512, kernel_size=3, stride=1, 'SAME')

	with tf.name_scope('pool4'):
		net = layer.max_pooling(net)

	with tf.name_scope('conv5'):
		net = layer.conv(net, input_chanel=512, num_output=512, kernel_size=3, stride=1, 'SAME')
		net = layer.conv(net, input_chanel=512, num_output=512, kernel_size=3, stride=1, 'SAME')
		net = layer.conv(net, input_chanel=512, num_output=512, kernel_size=3, stride=1, 'SAME')

	with tf.name_scope('pool5'):
		net = layer.max_pooling(net)

	with tf.name_scope('layer6'):
		_w = math.floor(w / 2 / 2 / 2 / 2 / 2)
		_h = math.floor(h / 2 / 2 / 2 / 2 / 2)
		net = tf.reshaped(x, [-1, _w * _h * 512])

		net = layer.fc(net, size=_w * _h * 512, num_output=4096)
		net = layer.dropout(net, keep_prob)

	with tf.name_scope('layer7'):
		net = layer.fc(net, size=4096, num_output=4096)
		net = layer.dropout(net, keep_prob)

	with tf.name_scope('output'):
		W_output = variable.init_weight([4096, 1000])
		b_output = variable.init_bias([1000])

		y = tf.matmual(net, W_output) + b_output
	
	return y, keep_prob