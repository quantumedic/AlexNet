import tensorflow as tf
import layer

def build(input, input_chanel, num_output):
	section1 = layer.conv(input, input_chanel=input_chanel, num_output=64, kernel_size=1, stride=1, 'SAME')

	section2 = layer.conv(input, input_chanel=input_chanel, num_output=96, kernel_size=1, stride=1, 'SAME')
	section2 = layer.conv(section2, input_chanel=96, num_output=128, kernel_size=3, stride=1, 'SAME')

	section3 = layer.conv(input, input_chanel=input_chanel, num_output=16, kernel_size=1, stride=1, 'SAME')
	section3 = layer.conv(input, input_chanel=16, num_output=32, kernel_size=5, stride=1, 'SAME')

	section4 = layer.max_pooling(input)
	section4 = layer.conv(input, input_chanel=input_chanel, num_output=32, kernel_size=1, stride=1, 'SAME')

	return tf.concat([section1, section2, section3, section4], 3)