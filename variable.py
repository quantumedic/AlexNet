import tensorflow as tf

def init_weight(shape):
	weight = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(weight)

def init_bias(shape):
	bias = tf.constant(0.1, shape=shape)
	return tf.Variable(bias)