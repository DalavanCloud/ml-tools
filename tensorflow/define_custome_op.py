import tensorflow as tf
from tensorflow.python.framework import function                                                           

@function.Defun(tf.float32, tf.float32)
def bprop(x, dy):
  return x + dy

@function.Defun(tf.float32, grad_func=bprop)
def fprop(x):
  return x  # identity

a = tf.Variable(tf.constant([-5., 4., -3., 2., 1.], dtype=tf.float32))
grad = tf.gradients(fprop(a), [a])                                         

with tf.Session() as sess:                                                             
  sess.run(tf.initialize_all_variables())
  result = sess.run(grad)                                                        

print(result)                                                                  