import math
import numpy as np 
import tensorflow as tf


from tensorflow.python.framework import ops

from utils import load_data,load_image,preprocess_A_and_B,get_image,save_images,imread,merge_images,merge,imsave,inverse_transform

class batch_norm(object):
            # h1 = lrelu(tf.contrib.layers.batch_norm(conv2d(h0, self.df_dim*2, name='d_h1_conv'),decay=0.9,updates_collections=None,epsilon=0.00001,scale=True,scope="d_h1_conv"))
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        # return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon, scale=True, scope=self.name)
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon, scale=True, scope=self.name)
def BN(x,epsilon=1e-5, momentum = 0.9, name="batch_norm",is_training=True):
    with tf.variable_scope(name):
        # bn=tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon, scale=True, name=name)
        bn=tf.contrib.layers.batch_norm(x, decay=momentum, updates_collections=None, epsilon=epsilon, scale=True,is_training=is_training, scope=name)
        return bn

def binary_cross_entropy(preds, targets, name=None):
    """Computes binary cross entropy given `preds`.

    For brevity, let `x = `, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        preds: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `preds`.
    """
    eps = 1e-12
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) +
                              (1. - targets) * tf.log(1. - preds + eps)))

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def conv2d_my(input, output_dim,k_size, d_size, stddev=0.02,
           name="conv2d_my"):
    with tf.variable_scope(name):


        w = tf.get_variable('w', [k_size, k_size, input.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input, w, strides=[1, d_size, d_size, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        tf.nn.bias_add(conv, biases)
        # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv       
def conv2d_atrous(input,output_dim,k_size,rate,stddev=0.02,name='conv2d_atrous'):
    with tf.variable_scope(name):
        w=tf.get_variable('w',[k_size, k_size, input.get_shape()[-1], output_dim],initializer=tf.truncated_normal_initializer(stddev=stddev))

        conv_atrous=tf.nn.atrous_conv2d(input,w,rate,padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        tf.nn.bias_add(conv_atrous, biases)
        return conv_atrous

def aspp(input,k_size,name='aspp',is_training=True):
    with tf.variable_scope(name):
        dims=input.shape[-1]
        
        f1=tf.nn.relu(BN(conv2d_atrous(input,dims,k_size,rate=1,name='f1'),name='bnf1',is_training=is_training))
        f2=tf.nn.relu(BN(conv2d_atrous(input,dims,k_size,rate=3,name='f2'),name='bnf2',is_training=is_training))
        f3=tf.nn.relu(BN(conv2d_atrous(input,dims,k_size,rate=5,name='f3'),name='bnf3',is_training=is_training))
        f4=tf.nn.relu(BN(conv2d_atrous(input,dims,k_size,rate=7,name='f4'),name='bnf4',is_training=is_training))
        f1_=BN(conv2d_my(f1,1,1,1,name='convf1_'),name='bnf1_',is_training=is_training)
        f2_=BN(conv2d_my(f2,1,1,1,name='convf2_'),name='bnf2_',is_training=is_training)
        f3_=BN(conv2d_my(f3,1,1,1,name='convf3_'),name='bnf3_',is_training=is_training)
        f4_=BN(conv2d_my(f4,1,1,1,name='convf4_'),name='bnf4_',is_training=is_training)
        d_concat=tf.nn.relu(conv2d_my((tf.concat([f1_,f2_,f3_,f4_],3)),4,3,1,name="convatt3"))
        
        # W=tf.nn.relu(conv2d_my(d_concat,d_concat.shape[-1],3,1,name='convw1'))
        W_=tf.nn.softmax(conv2d_my(d_concat,4,1,1,name='convw2'))
        w1=tf.expand_dims(W_[:,:,:,0],-1)
        w2=tf.expand_dims(W_[:,:,:,1],-1)
        w3=tf.expand_dims(W_[:,:,:,2],-1)
        w4=tf.expand_dims(W_[:,:,:,3],-1)
        F1=w1*f1_
        F2=w2*f2_
        F3=w3*f3_
        F4=w4*f4_
        F=F1+F2+F3+F4
        return F

def Context_Guided_Module(input, output_dim, name="cgm"):
    with tf.variable_scope(name):

        conv=conv2d_(input,output_dim,name='cgm_conv')
        conv_atrous=conv2d_atrous(tf.concat([input,conv],3),2*output_dim,3,2,name='cgm_atrous')
        output=conv2d_(tf.concat([input,conv,conv_atrous],3),2*output_dim,1,1,name='cgm_conv1')

        return output 
       
def conv2d(input_, output_dim, 
           k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        tf.nn.bias_add(conv, biases)
        # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv



def bilinear_sample(input_, x, y):
    H = tf.shape(input_)[1]
    W = tf.shape(input_)[2]

    x0 = tf.cast(tf.floor(x), tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), tf.int32)
    y1 = y0 + 1

    x0 = tf.clip_by_value(x0, 0, W - 1)
    x1 = tf.clip_by_value(x1, 0, W - 1)
    y0 = tf.clip_by_value(y0, 0, H - 1)
    y1 = tf.clip_by_value(y1, 0, H - 1)

    Ia = tf.gather_nd(input_, tf.stack([y0, x0], axis=-1))
    Ib = tf.gather_nd(input_, tf.stack([y1, x0], axis=-1))
    Ic = tf.gather_nd(input_, tf.stack([y0, x1], axis=-1))
    Id = tf.gather_nd(input_, tf.stack([y1, x1], axis=-1))

    wa = (tf.cast(x1, tf.float32) - x) * (tf.cast(y1, tf.float32) - y)
    wb = (tf.cast(x1, tf.float32) - x) * (y - tf.cast(y0, tf.float32))
    wc = (x - tf.cast(x0, tf.float32)) * (tf.cast(y1, tf.float32) - y)
    wd = (x - tf.cast(x0, tf.float32)) * (y - tf.cast(y0, tf.float32))

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


    
def conv2d_(input_, output_dim, 
           k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02,
           name="conv2d_"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        tf.nn.bias_add(conv, biases)
        # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def conv2d_1(input_, output_dim, 
           k_h=1, k_w=1, d_h=1, d_w=1, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        tf.nn.bias_add(conv, biases)
        # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

def deconv2d_(input_,output_shape,k_h=3,k_w=3,d_h=1,d_w=1,stddev=0.02,name="deconv2d_",with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_shape[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        
        try:
            upsample_image=tf.image.resize(input_,[output_shape[1],output_shape[2]],method="bilinear")
            deconv=tf.nn.conv2d(upsample_image,w,strides=[1,1,1,1],padding="SAME")
            # deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
            #                     strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        tf.nn.bias_add(deconv, biases)
        # deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def deconv2d_my(input_,
             k_size, d_size,  stddev=0.02,
             name="deconv2d", with_w=False):
    shape=input_.get_shape().as_list()
    re_img=tf.image.resize_images(input_,[2*shape[1],2*shape[2]],method=0)
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_size, k_size, shape[-1],(shape[-1])/2 ],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))  
        deconv=tf.nn.conv2d(re_img,w,strides=[1,d_size,d_size,1],padding='SAME') 
        biases = tf.get_variable('biases', [(shape[-1])/2], initializer=tf.constant_initializer(0.0))
        tf.nn.bias_add(deconv, biases)
        if with_w:
            return deconv, w, biases
        else:
            return deconv        

def deconv2d(input_, output_shape,
             k_size, d_size,  stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_size, k_size, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_size, d_size, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, d_size, d_size, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        tf.nn.bias_add(deconv, biases)
        # deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv
    

def deconv2d_(input_, output_shape,
             k_h=5, k_w=5, d_h=3, d_w=3, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        tf.nn.bias_add(deconv, biases)
        # deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv    

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
def atten(input_g,input_x,output_dim,name = "attention"):
    
    g = tf.contrib.resampler(input_g)
    
    with tf.variable_scope(name):
        
        w1 = tf.get_variable('w1', [1, 1, g.get_shape()[-1], g.get_shape()[-1]/2],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))        
        conv_g = tf.nn.conv2d(g, w1, strides=[1, 1, 1, 1], padding='SAME')
        biases1 = tf.get_variable('biases1', [g.get_shape()[-1]/2], initializer=tf.constant_initializer(0.0))        
        conv_g = tf.reshape(tf.nn.bias_add(conv_g, biases1), conv_g.get_shape())
        
        w2 = tf.get_variable('w2', [1, 1, input_x.get_shape()[-1], input_x.get_shape()[-1]/2],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))        
        conv_x = tf.nn.conv2d(input_x, w2, strides=[1, 1, 1, 1], padding='SAME')
        biases2 = tf.get_variable('biases2', [input_x.get_shape()[-1]/2], initializer=tf.constant_initializer(0.0))
        conv_x = tf.reshape(tf.nn.bias_add(conv_x, biases2), conv_x.get_shape())
    


def compute_gradient_penalty(discriminator, real_images, generated_images):
    batch_size = tf.shape(real_images)[0]

    # Calculate interpolation points between real images and generated images
    alpha = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
    interpolated_images = real_images + alpha * (generated_images - real_images)

    # Calculate gradients of discriminator output with respect to interpolated images
    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        interpolated,interpolated_output = discriminator(interpolated_images,reuse=True)
    interpolated_gradients = tape.gradient(interpolated_output, interpolated_images)

    # Calculate magnitude of gradients and take average over batch
    gradient_norms = tf.sqrt(tf.reduce_sum(tf.square(interpolated_gradients), axis=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean((gradient_norms - 1.) ** 2)

    return gradient_penalty


def wgan_gp_loss(discriminator, real_images, generated_images):
    # Calculate Wasserstein distance between real and generated images
    real,real_output = discriminator(real_images,reuse=True)
    gen,gen_output = discriminator(generated_images,reuse=True)
    wasserstein_distance = tf.reduce_mean(gen_output) - tf.reduce_mean(real_output)

    # Calculate gradient penalty
    gradient_penalty = compute_gradient_penalty(discriminator, real_images, generated_images)

    # Define total loss as sum of Wasserstein distance and gradient penalty
    wgan_gp_loss = wasserstein_distance + 10. * gradient_penalty
    
    return wgan_gp_loss


def cbam_block(input_feature, name, ratio=8):

    with tf.variable_scope(name):
        attention_feature = channel_attention(input_feature, 'channel_attention', ratio)
        attention_feature = spatial_attention(attention_feature, 'spatial_attention')
        print("CBAM Block Applied")
    return attention_feature

def channel_attention(input_feature, name, ratio=8):
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)
  
    with tf.variable_scope(name):
    
        channel = input_feature.get_shape()[-1]
        avg_pool = tf.reduce_mean(input_feature, axis=[1,2], keepdims=True)
        
        assert avg_pool.get_shape()[1:] == (1,1,channel)
        avg_pool = tf.layers.dense(inputs=avg_pool,
                                    units=channel//ratio,
                                    activation=tf.nn.relu,
                                    kernel_initializer=kernel_initializer,
                                    bias_initializer=bias_initializer,
                                    name='mlp_0',
                                    reuse=None)   
        assert avg_pool.get_shape()[1:] == (1,1,channel//ratio)
        avg_pool = tf.layers.dense(inputs=avg_pool,
                                    units=channel,                             
                                    kernel_initializer=kernel_initializer,
                                    bias_initializer=bias_initializer,
                                    name='mlp_1',
                                    reuse=None)    
        assert avg_pool.get_shape()[1:] == (1,1,channel)

        max_pool = tf.reduce_max(input_feature, axis=[1,2], keepdims=True)    
        assert max_pool.get_shape()[1:] == (1,1,channel)
        max_pool = tf.layers.dense(inputs=max_pool,
                                    units=channel//ratio,
                                    activation=tf.nn.relu,
                                    name='mlp_0',
                                    reuse=True)   
        assert max_pool.get_shape()[1:] == (1,1,channel//ratio)
        max_pool = tf.layers.dense(inputs=max_pool,
                                    units=channel,                             
                                    name='mlp_1',
                                    reuse=True)  
        assert max_pool.get_shape()[1:] == (1,1,channel)

        scale = tf.sigmoid(avg_pool + max_pool, 'sigmoid')
    
    return input_feature * scale

def spatial_attention(input_feature, name):

    kernel_size = 7
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    with tf.variable_scope(name):
        avg_pool = tf.reduce_mean(input_feature, axis=[3], keepdims=True)
        assert avg_pool.get_shape()[-1] == 1
        max_pool = tf.reduce_max(input_feature, axis=[3], keepdims=True)
        assert max_pool.get_shape()[-1] == 1
        concat = tf.concat([avg_pool,max_pool], 3)
        assert concat.get_shape()[-1] == 2
     
        concat = tf.layers.conv2d(concat,
                                filters=1,
                                kernel_size=[kernel_size,kernel_size],
                                strides=[1,1],
                                padding="same",
                                activation=None,
                                kernel_initializer=kernel_initializer,
                                use_bias=False,
                                name='conv')
        assert concat.get_shape()[-1] == 1
        concat = tf.sigmoid(concat, 'sigmoid')
    
    return input_feature * concat

def se_block(residual, name, ratio=8):
#       """Contains the implementation of Squeeze-and-Excitation(SE) block.
#   As described in https://arxiv.org/abs/1709.01507.
#   """
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)

    with tf.variable_scope(name):
        channel = residual.get_shape()[-1]
        # Global average pooling
        squeeze = tf.reduce_mean(residual, axis=[1,2], keepdims=True)
        assert squeeze.get_shape()[1:] == (1,1,channel)
        excitation = tf.layers.dense(inputs=squeeze,
                                     units=channel//ratio,
                                     activation=tf.nn.relu,
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     name='bottleneck_fc')
        assert excitation.get_shape()[1:] == (1,1,channel//ratio)
        excitation = tf.layers.dense(inputs=excitation,
                                     units=channel,
                                     activation=tf.nn.sigmoid,
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     name='recover_fc')
        assert excitation.get_shape()[1:] == (1,1,channel)
        # top = tf.multiply(bottom, se, name='scale')
        scale = residual * excitation
    return scale


def conv_layer(input, filters, kernel_size, strides, padding, name):
    with tf.variable_scope(name):
        conv = tf.layers.conv2d(input, filters, kernel_size, strides, padding, 
                                kernel_initializer=tf.glorot_uniform_initializer())
        conv = tf.layers.batch_normalization(conv)
        conv = tf.nn.relu(conv)
    return conv

def attention_layer(input, name):
    with tf.variable_scope(name):
        # Global average pooling to generate attention maps
        gap = tf.reduce_mean(input, axis=[1, 2], keepdims=True)  # Global Average Pooling
        attention = tf.layers.dense(gap, units=input.get_shape()[-1], activation=tf.nn.sigmoid)
        return input * attention

def multi_scale_attention(input, name):
    with tf.variable_scope(name):
        conv_1 = conv_layer(input, filters=8, kernel_size=1, strides=1, padding='same', name='conv_1x1')
        conv_3 = conv_layer(input, filters=8, kernel_size=3, strides=1, padding='same', name='conv_3x3')
        conv_5 = conv_layer(input, filters=8, kernel_size=5, strides=1, padding='same', name='conv_5x5')

        attention_1 = attention_layer(conv_1, 'attention_1x1')
        attention_3 = attention_layer(conv_3, 'attention_3x3')
        attention_5 = attention_layer(conv_5, 'attention_5x5')

        multi_scale_features = tf.concat([attention_1, attention_3, attention_5], axis=-1)
    return multi_scale_features



def dice_loss(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    dice = (2.000 * intersection + smooth) / (union + smooth)
    return 1 - tf.reduce_mean(dice)


def dice_loss_weight(y_true, y_pred,weights, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred * weights, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true * weights, axis=[1, 2, 3]) + tf.reduce_sum(y_pred * weights, axis=[1, 2, 3])
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - tf.reduce_mean(dice)


def tversky_loss(y_true, y_pred, smooth=1, alpha=0.7, beta=0.3):
    true_pos = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    false_neg = tf.reduce_sum(y_true * (1 - y_pred), axis=[1, 2, 3])
    false_pos = tf.reduce_sum((1 - y_true) * y_pred, axis=[1, 2, 3])
    tversky = (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)
    return 1 - tf.reduce_mean(tversky)

def weighted_dice_loss(pred, target, class_weights):
    smooth = 1.0
    
    intersection = tf.reduce_sum(pred * target, axis=[1, 2, 3])  # 在空间维度上求和
    pred_sum = tf.reduce_sum(pred, axis=[1, 2, 3])
    target_sum = tf.reduce_sum(target, axis=[1, 2, 3])

    dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)

    class_weights_expanded = tf.reshape(class_weights, [1, 1, 1, -1])  # 形状变为 [1, 1, 1, 5]
    class_weights_expanded = tf.tile(class_weights_expanded, [tf.shape(pred)[0], tf.shape(pred)[1], tf.shape(pred)[2], 1])  # 广播到 [batch_size, height, width, 5]

    weighted_dice = tf.reduce_sum(class_weights_expanded * dice, axis=[1, 2, 3])  # 按类别加权

    return 1 - weighted_dice





