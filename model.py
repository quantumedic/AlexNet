import tensorflow as tf
import math
import layer

def alexNet(x, w, h):
	keep_prob = tf.placeholder(tf.float32)

	with tf.name_scope('reshape'):
		net = tf.reshaped(x, [-1, w, h, 3])

	with tf.name_scope('layer1'):
		net = layer.conv(net, input_chanel=3, num_output=96, kernel_size=11, stride=4, 'VALID')
		net = layer.max_pooling(net)
		net = layer.lrn(net)

	with tf.name_scope('layer2'):
		net = layer.conv(net, input_chanel=96, num_output=256, kernel_size=5, stride=1, 'VALID')
		net = layer.max_pooling(net)
		net = layer.lrn(net)

	with tf.name_scope('layer3'):
		net = layer.conv(net, input_chanel=256, num_output=384, kernel_size=3, stride=1, 'SAME')

	with tf.name_scope('layer4'):
		net = layer.conv(net, input_chanel=384, num_output=384, kernel_size=3, stride=1, 'SAME')
		
	with tf.name_scope('layer5'):
		net = layer.conv(net, input_chanel=384, num_output=256, kernel_size=3, stride=1, 'SAME')
		net = layer.max_pooling(net)

	with tf.name_scope('layer6'):
		_w = math.floor(w / 4 / 2 / 2 / 2)
		_h = math.floor(h / 4 / 2 / 2 / 2)
		net = tf.reshaped(x, [-1, _w * _h * 256])

		net = layer.fc(net, size=_w * _h * 256, num_output=4096)
		net = layer.dropout(net, keep_prob)

	with tf.name_scope('layer7'):
		net = layer.fc(net, size=4096, num_output=4096)
		net = layer.dropout(net, keep_prob)

	with tf.name_scope('output'):
		W_output = variable.init_weight([4096, 1000])
		b_output = variable.init_bias([1000])

		y = tf.matmual(net, W_output) + b_output
	
	return y, keep_prob