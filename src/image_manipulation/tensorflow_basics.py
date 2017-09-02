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