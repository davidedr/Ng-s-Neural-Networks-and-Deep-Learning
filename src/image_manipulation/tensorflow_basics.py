'''
Created on 02 set 2017

@author: davide
'''

import numpy as np
x = np.linspace(-3.0, 3.0, 100)
print(len(x))
print(type(x))
print(x.dtype)
print(x.shape)
print(x)

print()
print('USING TENSOR FLOW')
print()

import tensorflow as tf
print('Get default graph...')
g = tf.get_default_graph()
print('Print operations...')
[op.name for op in g.get_operations()]

print('Get linspace...')
x = tf.linspace(-3.0, 3.0, 100)
print(type(x))
print(x.dtype)
print(x.shape)
print(x)

print('Get default graph...')
g = tf.get_default_graph()
print('Print operations...')
[op.name for op in g.get_operations()]

name = 'LinSpace' + ':0'
print("Get tensor by name: '" + name + "'...")
g.get_tensor_by_name(name)
print(g)
      
print('Creating session...')
s = tf.Session()
print('Running session...')      
computed_x = s.run(x)
print(computed_x)
s.close()

print('Another way to perform the same computation:')
s = tf.Session()
computed_x = x.eval(session = s)
print(computed_x)
s.close()

print('Generate and plot a Gaussian...')
# Keep in mind that in tf.pow(x, y) BOTH arguments MUST be of the same type!
# Both integers, both float, ...
mean = 0.0   #float!
stddev = 1.0 #float!
z = (tf.exp(tf.negative(tf.pow(x - mean, 2.0) /
                   (2.0 * tf.pow(stddev, 2.0)))) *
     (1.0 / (stddev * tf.sqrt(2.0 * 3.1415))))
s = tf.Session()
result = z.eval(session = s)
s.close()

%matplotlib inline
import matplotlib.pyplot as plt
plt.plot(result)

print('Generate and plot a 2D Gaussian...')
ksize = z.get_shape().as_list()[0]
# Clever way to get a 2-D Gaussian: multiply a 1 x k vectot by its traspose!
z_2d = tf.matmul(tf.reshape(z, [ksize, 1]), tf.reshape(z, [1, ksize]))

s = tf.Session()
result = z_2d.eval(session = s)
s.close()

# A clever way to plot it: see it as an image!
plt.imshow(result)
