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

##
## Images convolution with Gaussian kernels
##

from skimage import data
img = data.camera().astype(np.float32)
plt.imshow(img, cmap='gray')
plt.show()
print('Image shape: ' + str(img.shape))

image_4d = img.reshape([1, img.shape[0], img.shape[1], 1])
print('image_4d shape: ' + str(img.shape))

# OR use tf reshape capability:
img_4d = tf.reshape(img, [1, img.shape[0], img.shape[1], 1])
print(img_4d)
print(img_4d.get_shape())
print(img_4d.get_shape().as_list())

# Reshape the image to N x W x H x C = 1 x W x H x 1 in order to be able to convolve it w/ gaussian kernel
# Only N = 1 images, only C = 1 channels

# via reshape
img_4d = img.reshape([1, img.shape[0], img.shape[1], 1])
print('img_4d.shape (via img.reshape): ' + str(img_4d.shape))

# via tensorflow
img_4d = tf.reshape(img, [1, img.shape[0], img.shape[1], 1])
print('img_4d.shape (via tf.reshape): ' + str(img_4d.shape))

# Reshape the kernel to Kernel Height x Kernel Width x Number of Input Channels x Number of Output Channels
# Number of Input Channels := C = 1
# Number of Output Channels := Number of Input Channels
z_4d = tf.reshape(z_2d, [ksize, ksize, 1, 1])
print('z_4d.shape (via tf.reshape): ' + str(z_4d.shape))

# Perform convolution
convolved = tf.nn.conv2d(img_4d, z_4d, strides = [1, 1, 1, 1], padding = 'SAME')

s = tf.Session()
img_convolved = convolved.eval(session = s)
s.close()

print('img_convolved.shape: ' + str(img_convolved.shape))

# Plot result
plt.figure()
plt.imshow(img_convolved[0, :, :, 0], cmap = 'gray')
plt.show()

#
# Gabor kernel: Gaussian kernel modulated with a Sine wave
#
xs = tf.linspace(-3.0, 3.0, ksize)
ys = tf.sin(xs)

plt.figure()
s = tf.Session()
ys = ys.eval(session = s)
s.close()

plt.plot(ys)
plt.show()

ys = tf.reshape(ys, [ksize, 1])
ones = tf.ones((1, ksize))
wave = tf.matmul(ys, ones)

s = tf.Session()
wave = wave.eval(session = s)
s.close()

plt.figure()
plt.imshow(wave, cmap = 'gray')
plt.show()

gabor_kernel = tf.multiply(wave, z_2d)

s = tf.Session()
gabor_kernel = gabor_kernel.eval(session = s)
s.close()

plt.figure()
plt.imshow(gabor_kernel, cmap = 'gray')
plt.show()

#
# Computing and using the Gabor kernel via Tensorflow placeholders
#
img = tf.placeholder(tf.float32, shape = [None, None], name = 'img')

# Add a channel H x W -->> H x W x 1
img_3d = tf.expand_dims(img, 2)
print('img_3d.get_shape: ' + str(img_3d.get_shape()))

# H x W x 1 -->> 1 x H x W x 1
img_4d = tf.expand_dims(img_3d, 0)
print('img_4d.get_shape: ' + str(img_4d.get_shape()))

# Placeholders for the Gabor kernel parameters
mean = tf.placeholder(tf.float32, name = 'mean')
sigma = tf.placeholder(tf.float32, name = 'sigma')
ksize = tf.placeholder(tf.int32, name = 'ksize')
                  
# Generate the tensor flow graph to compute the convolution w/ the Gabor kernel
x = tf.linspace(-3.0, 3.0, ksize)
z = (tf.exp(tf.negative(tf.pow(x - mean, 2.0) /
                   (2.0 * tf.pow(sigma, 2.0)))) *
      (1.0 / (sigma * tf.sqrt(2.0 * 3.1415))))
z_2d = tf.matmul(
  tf.reshape(z, tf.stack([ksize, 1])),
  tf.reshape(z, tf.stack([1, ksize])))
ys = tf.sin(x)
ys = tf.reshape(ys, tf.stack([ksize, 1]))
ones = tf.ones(tf.stack([1, ksize]))
wave = tf.matmul(ys, ones)
gabor = tf.multiply(wave, z_2d)
gabor_4d = tf.reshape(gabor, tf.stack([ksize, ksize, 1, 1]))

# And finally, convolve the two:
convolved = tf.nn.conv2d(img_4d, gabor_4d, strides=[1, 1, 1, 1], padding='SAME', name='convolved')
convolved_img = convolved[0, :, :, 0]

# Perform actual computation feeding the placeholders' values
s = tf.Session()
res = convolved_img.eval(session = s, feed_dict = { img: data.camera(), mean: 0.0, sigma: 1.0, ksize: 100 })
s.close()

plt.figure()
plt.imshow(res, cmap = 'gray')
plt.show()

# Once the TF graph is set up, we can repeat the computation using different parameters

s = tf.Session()
res = convolved_img.eval(session = s, feed_dict={
    img: data.camera(),
    mean: 0.0,
    sigma: 0.5,
    ksize: 32
  })
s.close()

plt.figure()
plt.imshow(res, cmap = 'gray')
plt.show()
