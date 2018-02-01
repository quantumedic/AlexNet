import tensorflow as tf
import variable

def conv(input, input_chanel, num_output, kernel_size, stride, padding):
	W_conv = variable.init_weight([kernel_size, kernel_size, input_chanel, num_output])
	b_conv = variable.init_bias([num_output])

	conv_layer = tf.nn.conv2d(
		input,
		W_conv,
		strides=[1, stride, stride, 1],
		padding=padding)
	return tf.nn.relu(conv_layer + b_conv)

def max_pooling(input):
	return tf.nn.max_pooling(
		input,
		ksize=[1, 3, 3, 1],
		strides=[1, 2, 2, 1],
		padding='SAME')

def lrn(input):
	return tf.nn.local_response_normalization(input, 5)

def dropout(input, keep_prob):
	return tf.nn.dropout(input, keep_prob)

def fc(input, size, num_output):
	W_fc = variable.init_weight([size, num_output])
	b_fc = variable.init_bias([num_output])

	fc_layer = tf.matmual(input, W_fc) + b_fc
	return tf.nn.relu(fc_layer)