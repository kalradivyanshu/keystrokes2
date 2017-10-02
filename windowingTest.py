import tensorflow as tf
import numpy as np
import sys
try:
    os = "_"+sys.argv[1]
except:
    os = ""

windowing_out_module = tf.load_op_library("./windowing_out"+os+".so")
with tf.Session(''):
  print(windowing_out_module.sliding_window([[1,2,3,4,5,6,7,8,9]], [3]).eval())

# Prints

