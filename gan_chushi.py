from __future__ import division
import math
import os

import time
from glob import glob
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 


from ops import conv2d,lrelu,conv2d_,deconv2d_,conv2d_1,deconv2d_my,wgan_gp_loss,BN,Context_Guided_Module
from utils import load_data,load_image,save_images,imread,merge,imsave,save_mat

class pix2pix_chushi(object):
    def __init__(self, sess, image_size=384,
                 batch_size=2, sample_size=1, output_size=256,
                 gf_dim=32, df_dim=32, L1_lambda=20,
                 input_c_dim=1, output_c_dim=8, dataset_name='facades',
                 checkpoint_dir=None, sample_dir=None
                 ):
        
        #tf.reset_default_graph()
        self.sess = sess
        self.is_grayscale = (input_c_dim == 1)
        self.batch_size = batch_size
 #       self.batch_size = 1
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim

        self.L1_lambda = L1_lambda
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):
#        self.real_data = tf.placeholder(tf.float32,
#                                        [self.batch_size, self.image_size, self.image_size,
#                                         self.input_c_dim + self.output_c_dim],
#                                        name='real_A_and_B_images')
        self.real_data = tf.placeholder(tf.float32,
                                        [None, self.image_size, self.image_size,
                                         self.input_c_dim + self.output_c_dim],
                                        name='real_A_and_B_images')
        self.is_training=tf.placeholder(tf.bool,name="is_training")
        self.dropout_rate=tf.placeholder(tf.float32,name='dropout_rate')
        
        self.real_B = self.real_data[:, :, :, :self.output_c_dim]
        self.real_A = self.real_data[:, :, :, self.output_c_dim:self.input_c_dim + self.output_c_dim]

        self.fake_B = self.generator(self.real_A)
        #D_logits：D(x|y)，D_logits_：D(G(x)|y)
        self.real_AB = tf.concat([self.real_A, self.real_B], 3)
        self.fake_AB = tf.concat([self.real_A, self.fake_B], 3)
        
        self.D, self.D_logits = self.discriminator(self.real_AB,is_training=self.is_training ,reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.fake_AB, is_training=self.is_training,reuse=True)

        self.fake_B_sample = self.sampler(self.real_A,is_training=self.is_training,dropout_rate=self.dropout_rate)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.fake_B_sum = tf.summary.image("fake_B", self.fake_B)

        
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
#        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_))) \
#                        + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B - self.fake_B))
        
        self.g_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))
        self.g_loss2 = self.L1_lambda * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.fake_B, labels=self.real_B))+ tf.reduce_mean(tf.abs(self.real_B - self.fake_B))
#        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_))) 
        
        # # class_weights = tf.constant([[ 1.3868,0.3602,26.2978, 8.5811 ,0.2300 ,0.2097,7.0093,23.2234,0.0489]])
        # class_weights = tf.constant([[ 1., 0.30547066, 3.60645148 , 7.24898929, 0.09778477, 0.14040097, 1.61237653,  16.78074101, 0.0294764]])
        # flat_logits = tf.reshape(self.fake_B, [-1, self.output_c_dim])#[N,n_class]
        # flat_labels = tf.reshape(self.real_B, [-1, self.output_c_dim])#[N,n_class]
        # weight_map = tf.multiply(flat_labels, class_weights)
        # weight_map = tf.reduce_sum(weight_map, axis=1)
        # loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)
        # weighted_loss = tf.multiply(loss_map, weight_map)
        # self.seg_ce = tf.reduce_mean(weighted_loss)
        self.g_loss = self.g_loss1+self.g_loss2
        # self.g_loss = self.g_loss1+self.L1_lambda *self.seg_ce
        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake
        # self.g_loss = -tf.reduce_mean(self.D_logits_) #生成器loss
        # self.d_loss = tf.reduce_mean(self.D_logits_) - tf.reduce_mean(self.D_logits) #判别器loss
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]


        self.saver = tf.train.Saver(max_to_keep=0)


    def load_random_samples(self):
        data_=[]
        for root,dirs,files in os.walk('E:/zys/GAN/datasets/{}/test'.format(self.dataset_name),topdown='False'):
            for dir in dirs:
                dirs_path=os.path.join(root,dir)
                imgs=os.listdir(dirs_path)
                for img in imgs:
                    img_path=dirs_path+'/'+img
                    data_.append(img_path)    
        data = np.random.choice(data_, self.batch_size)
        sample = [load_data(sample_file) for sample_file in data]
        if (self.is_grayscale):
            sample_images = np.array(sample).astype(np.float32)
        else:
            sample_images = np.array(sample).astype(np.float32)
        return sample_images
    
    def load_train_samples(self):
        sample_data = glob('E:/zys/GAN/datasets/{}/trainsample/*.mat'.format(self.dataset_name))
        
        sample =  [load_data(sample_file) for sample_file in sample_data]
        if (self.is_grayscale):
            sample_images = np.array(sample).astype(np.float32)
        else:
            sample_images = np.array(sample).astype(np.float32)
        return sample_images
    
    
    def sample_model2(self,sample_dir,sample_dir_, sample_images,sample_images_, epoch, idx):
        samples, d_loss, g_loss = self.sess.run(
            [self.fake_B_sample, self.d_loss, self.g_loss],
            feed_dict={self.real_data: sample_images,self.is_training:False,self.dropout_rate:1.0}
        )
        samples_, d_loss_, g_loss_ = self.sess.run(
            [self.fake_B_sample, self.d_loss, self.g_loss],
            feed_dict={self.real_data: sample_images_,self.is_training:False,self.dropout_rate:1.0}
        )
        

        save_images(samples,sample_images,[self.batch_size, 9],
                    '{}/train_{:02d}_{:04d}.bmp'.format(sample_dir, epoch, idx))
        save_images(samples_,sample_images_, [self.batch_size, 9],
                    '{}/T_{:02d}_{:04d}.bmp'.format(sample_dir_, epoch, idx))
        print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))


    def sample_model(self, sample_dir,sample_dir_, epoch, idx):
        sample_images = self.load_random_samples()
        sample_images_ = self.load_train_samples()
        samples, d_loss, g_loss = self.sess.run(
            [self.fake_B_sample, self.d_loss, self.g_loss],
            feed_dict={self.real_data: sample_images}
        )
        samples_, d_loss_, g_loss_ = self.sess.run(
            [self.fake_B_sample, self.d_loss, self.g_loss],
            feed_dict={self.real_data: sample_images_}
        )
            
        save_images(samples,sample_images,[self.batch_size, 9],
                    '{}/train_{:02d}_{:04d}.bmp'.format(sample_dir, epoch, idx))
        save_images(samples_,sample_images_, [self.batch_size, 9],
                    '{}/T_{:02d}_{:04d}.bmp'.format(sample_dir_, epoch, idx))
        print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

    def plot_loss(self,loss,save_name):  
        fig,ax = plt.subplots(figsize=(20,7))  
        losses = np.array(loss) 
        plt.plot(losses.T[0], label="Discriminator Loss")  
        plt.plot(losses.T[1], label="Generator Loss") 
        plt.title("Training Losses")  
        plt.legend()  
        plt.savefig(save_name)  
        plt.show() 

    def train(self, args):

        print('train')
        
        """Train pix2pix"""
        losses = []  
        val_losses=[]
        d_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        
        # g_optim = tf.train.AdamOptimizer(args.lr,beta1=0.5,beta2=0.9).minimize(self.g_loss,var_list=self.g_vars)  
        # d_optim = tf.train.AdamOptimizer(args.lr,beta1=0.5,beta2=0.9).minimize(self.d_loss,var_list=self.d_vars) 

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

#        self.g_sum = tf.summary.merge([self.d__sum,
#            self.fake_B_sum, self.d_loss_fake_sum, self.g_loss_sum])
#        self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
#        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        data=[]

        print('1')

        for root,dirs,files in os.walk('E:/zys/GAN/datasets/{}/train'.format(self.dataset_name),topdown='False'):
            for dir in dirs:
                dirs_path=os.path.join(root,dir)
                imgs=os.listdir(dirs_path)
                for img in imgs:
                    img_path=dirs_path+'/'+img
                    data.append(img_path)
        
        print('2')

        # validation_data=[]
        # for root,dirs,files in os.walk('./datasets/{}/each_test'.format(self.dataset_name),topdown='False'):
        #     for dir in dirs:
        #         dirs_path=os.path.join(root,dir)
        #         imgs=os.listdir(dirs_path)
        #         for img in imgs:
        #             img_path=dirs_path+'/'+img
        #             validation_data.append(img_path)        
        validation_data=glob('E:/zys/GAN/datasets/{}/val/*.mat'.format(self.dataset_name))
        
        print('3')

        for epoch in range(55,args.epoch):
            print(epoch)
        
            epochd_loss=0
            epochg_loss=0
            epochd_valloss=0
            epochg_valloss=0
            #data = glob('./datasets/{}/train/*.mat'.format(self.dataset_name))
            np.random.shuffle(data)
            batch_idxs = math.floor(len(data)// self.batch_size)

            print('4')
            print(batch_idxs)

            for idx in range(0, batch_idxs):
                batch_files = data[idx*self.batch_size:(idx+1)*self.batch_size]
                batch = [load_data(batch_file) for batch_file in batch_files]
                # validation_files=np.random.choice(validation_data,self.batch_size)
                validation_image=[load_data(validation_file) for validation_file in validation_data]                
                if (self.is_grayscale):
                    batch_images = np.array(batch).astype(np.float32)
                    validation_images= np.array(validation_image).astype(np.float32)
                else:
                    batch_images = np.array(batch).astype(np.float32)
                    validation_images=np.array(validation_image).astype(np.float32)
                # Update D network
#                print(len(batch_images[1]))
                _ = self.sess.run([d_optim], feed_dict={ self.real_data: batch_images ,self.is_training:True,self.dropout_rate:0.9})
                # _ = self.sess.run([d_optim], feed_dict={ self.real_data: batch_images })
                # _ = self.sess.run([d_optim], feed_dict={ self.real_data: batch_images })
#                _, summary_str = self.sess.run([d_optim, self.d_sum],
#                                               feed_dict={ self.real_data: batch })
#                self.writer.add_summary(summary_str, counter)

                # Update G network
                _ = self.sess.run([g_optim],
                                               feed_dict={ self.real_data: batch_images  ,self.is_training:True,self.dropout_rate:0.9})
                _ = self.sess.run([g_optim],
                                               feed_dict={ self.real_data: batch_images  ,self.is_training:True,self.dropout_rate:0.9})                                               
#                 _, summary_str = self.sess.run([g_optim, self.g_sum],
#                                               feed_dict={ self.real_data: batch_images })
#                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                # _= self.sess.run([g_optim],
                #                                feed_dict={ self.real_data: batch_images })
#                self.writer.add_summary(summary_str, counter)
                
               
                errD_fake = self.d_loss_fake.eval({self.real_data: batch_images,self.is_training:True,self.dropout_rate:0.9})
                errD_real = self.d_loss_real.eval({self.real_data: batch_images,self.is_training:True,self.dropout_rate:0.9})
                errG = self.g_loss.eval({self.real_data: batch_images,self.is_training:True,self.dropout_rate:0.9})
#                errG1 = self.g_loss1.eval({self.real_data: batch_images})
#                errG2 = self.g_loss2.eval({self.real_data: batch_images})
                errD_fake_val = self.d_loss_fake.eval({self.real_data: validation_images,self.is_training:False,self.dropout_rate:1.0})
                errD_real_val = self.d_loss_real.eval({self.real_data: validation_images,self.is_training:False,self.dropout_rate:1.0})
                errG_val = self.g_loss.eval({self.real_data: validation_images,self.is_training:False,self.dropout_rate:1.0})


                errD = self.d_loss.eval({self.real_data: batch_images,self.is_training:True,self.dropout_rate:0.9})

                # errG = self.g_loss.eval({self.real_data: batch_images})
# #                errG1 = self.g_loss1.eval({self.real_data: batch_images})
# #                errG2 = self.g_loss2.eval({self.real_data: batch_images})
                errD_val = self.d_loss.eval({self.real_data: validation_images,self.is_training:False,self.dropout_rate:1.0})

#                 errG_val = self.g_loss.eval({self.real_data: validation_images})


                counter += 1
                # if idx == batch_idxs-1:
                #     print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, valid_loss:%.8f, valig_loss:%.8f " \
                #           % (epoch, idx, batch_idxs,
                #              time.time() - start_time, errD_fake+errD_real, errG, errD_fake_val+errD_real_val, errG_val ))
                # losses.append((errD_fake+errD_real,errG))
                print(idx)
                if idx == batch_idxs-1:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, valid_loss:%.8f, valig_loss:%.8f " \
                          % (epoch, idx, batch_idxs,
                             time.time() - start_time, errD_fake+errD_real, errG, errD_fake_val+errD_real_val, errG_val ))
                # losses.append((errD_real+errD_fake,errG))
                # val_losses.append((errD_fake_val+errD_real_val,errG_val))
                epochd_loss=epochd_loss+errD_real+errD_fake
                epochg_loss=epochg_loss+errG
                epochd_valloss=epochd_valloss+errD_real_val+errD_fake_val
                epochg_valloss=epochg_valloss+errG_val
                print('5')
                # if np.mod(counter,336 ) == 1:
                #     self.sample_model(args.sample_dir,args.sample_dir_, epoch, idx)
                if np.mod(counter,batch_idxs+1) ==1:    #在每个epoch结束时执行sample_model2操作
                    self.sample_model2(args.sample_dir,args.sample_dir_,batch_images,validation_images, epoch, idx)
            epochd_loss=epochd_loss/batch_idxs
            epochg_loss=epochg_loss/batch_idxs
            epochd_valloss=epochd_valloss/batch_idxs
            epochg_valloss=epochg_valloss/batch_idxs
            losses.append((epochd_loss,epochg_loss))
            val_losses.append((epochd_valloss,epochg_valloss))
            self.save(args.checkpoint_dir, epoch)
        self.plot_loss(losses,'loss.jpg')
        self.plot_loss(val_losses,'val_loss.jpg')


    def   discriminator(self, image,  is_training=True, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:

            print('discriminator')

            # image is 256 x 256 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            image=tf.reshape(image, tf.stack([-1, self.image_size,self.image_size, self.output_c_dim+self.input_c_dim ]))
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            # h0 is (192 x 192 x self.df_dim)
            # h0_=lrelu(conv2d_(image192,self.df_dim,k_h=1,k_w=1,name='d_h0__conv'))
  

            h1 = lrelu(BN(conv2d(h0, self.df_dim*2, name='d_h1_conv'),name='d_bn1',is_training=is_training))
            # h1 is (96 x 96 x self.df_dim*2)
            # h1_=lrelu(conv2d_(image96,self.df_dim*2,k_h=1,k_w=1,name='d_h1__conv'))


            h2 = lrelu(BN(conv2d(h1, self.df_dim*4, name='d_h2_conv'),name='d_bn2',is_training=is_training))
            # h2 is (48 x 48 x self.df_dim*4)
            # h2_=lrelu(conv2d_(image48,self.df_dim*4,k_h=1,k_w=1,name='d_h2__conv'))

            h3 = lrelu(BN(conv2d(h2, self.df_dim*8,  name='d_h3_conv'),name='d_bn3',is_training=is_training))
            # h3 is (24x 24 x self.df_dim*8)
            h4 = lrelu(BN(conv2d(h3, 1, d_h=1, d_w=1,name='d_h4_conv'),name='d_bn4',is_training=is_training))
#            # h4 is (12 x 12 x 1)
            # h5 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4
            # return h4, h4
            

    def generator(self, image, is_training=True,dropout_rate=0.8,y=None):
        with tf.variable_scope("generator") as scope:
            print('generator')
            s = self.output_size
            s2, s4, s8 = int(s/2), int(s/4), int(s/8)
            image=tf.reshape(image, tf.stack([-1, self.image_size,self.image_size, self.input_c_dim ]))
            # image is (384 x 384 x input_c_dim)
            e1_ = BN(conv2d_(image, self.gf_dim, name='g_e1_conv_'),name='g_bn_1_',is_training=is_training)
            # e1_ is (384 x 384 x self.gf_dim)
            
            e1 = BN(conv2d(e1_, self.gf_dim, name='g_e1_conv'),name='g_bn_1',is_training=is_training)
            # e1 is (192 x 192 x self.gf_dim)
            e2_ = BN(conv2d_(lrelu(e1),self.gf_dim,name='g_cgm_e2_'),name='g_bn_2_',is_training=is_training)
            # e2_ = BN(conv2d_(lrelu(e1), self.gf_dim*2, name='g_e2_conv_'),name='g_bn_2_',is_training=is_training)
            # e2_ is (192 x 192 x self.gf_dim*2)
            
            
            e2 = BN(conv2d(lrelu(e2_), self.gf_dim*2, name='g_e2_conv'),name='g_bn_2',is_training=is_training)
            # e2 is (96 x 96 x self.gf_dim*2)
            e3_ = BN(conv2d_(lrelu(e2),2*self.gf_dim,name='g_cgm_e3_'),name='g_bn_3_',is_training=is_training)
            # e3_ = BN(conv2d_(lrelu(e2), self.gf_dim*4, name='g_e3_conv_'),name='g_bn_3_',is_training=is_training)
            # e3_ is (96 x 96 x self.gf_dim*4)
            
            e3 = BN(conv2d(lrelu(e3_), self.gf_dim*4, name='g_e3_conv'),name='g_bn_3',is_training=is_training)
            # e3 is (48 x 48 x self.gf_dim*4)
            e4_ = BN(conv2d_(lrelu(e3),4*self.gf_dim,name='g_cgm_e4_'),name='g_bn_4_',is_training=is_training)
            # e4_ = BN(conv2d_(lrelu(e3), self.gf_dim*8, name='g_e4_conv_'),name='g_bn_4_',is_training=is_training)
            # e4_ is (48 x 48 x self.gf_dim*8)
            
            e4 = BN(conv2d(lrelu(e4_), self.gf_dim*8, name='g_e4_conv'),name='g_bn_4',is_training=is_training)
            # e4 is (24 x 24 x self.gf_dim*8)
            e5_ = BN(conv2d_(lrelu(e4),8*self.gf_dim,name='g_cgm_e5_'),name='g_bn_5_',is_training=is_training)
            # e5_ = BN(conv2d_(lrelu(e4), self.gf_dim*16, name='g_e5_conv_'),name='g_bn_5_',is_training=is_training)
            # e5_ is (24 x 24 x self.gf_dim*16)

            """
            attention gate1
            """
            # g2 = tf.image.resize_images(lrelu(e5_),[48,48],method=0)
            # # g2 is (48 x 48 x self.gf_dim*16)
            # x2 = BN(conv2d_1(lrelu(e4_), self.gf_dim*4, name='g_x2_conv'),name='g_bn_x2',is_training=is_training)
            # # x2_ is (48 x 48 x self.gf_dim*4)
            # g2_conv = BN(conv2d_1(g2, self.gf_dim*4, name='g_g2_conv'),name='g_bn_g2_',is_training=is_training)
            # # g2_conv is (48 x 48 x self.gf_dim*4)
            # psi2 = lrelu(x2 + g2_conv)
            # psi2_conv = BN(conv2d_1(psi2, 1, name='g_p2_conv'),name='g_bn_p2',is_training=is_training)
            # # psi2_conv is (48 x 48 x 1)
            # psi2_conv = tf.nn.sigmoid(psi2_conv)
            # e4_ = e4_*psi2_conv
            """
            up_conv_block1
            """
            self.d2, self.d2_w, self.d2_b = deconv2d_my(tf.nn.relu(e5_),k_size=3,d_size=1,name='g_d2', with_w=True)
            d2 = tf.nn.dropout(BN(self.d2,name='g_bn_d2',is_training=is_training), dropout_rate)
            d2 = tf.concat([d2, e4_], 3)
            # d2 is (48 x 48 x self.gf_dim*16)
            """
            conv_block1
            """
            self.d2_0 = BN(conv2d_(lrelu(d2), self.gf_dim*8, name='g_d2_0_conv_'),name='g_bn_d2_0',is_training=is_training)
            self.d2_1 = BN(conv2d_(lrelu(self.d2_0), self.gf_dim*8, name='g_d2_1_conv'),name='g_bn_d2_1',is_training=is_training)
            # self.d2_1 is (48 x 48 x self.gf_dim*8)
            self.d2_out=conv2d_(self.d2_1,1,k_h=1,k_w=1,name='g_d2_out_conv')
            #self.d2_out is 48*48*1


            """
            attention gate2
            """
            # g3 = tf.image.resize_images(lrelu(self.d2_1),[96,96],method=0)
            # # g3 is (96 x 96 x self.gf_dim*4)
            # x3 = BN(conv2d_1(lrelu(e3_), self.gf_dim*2, name='g_x3_conv'),name='g_bn_x3',is_training=is_training)
            # # x3_ is (96 x 96 x self.gf_dim*2)
            # g3_conv = BN(conv2d_1(g3, self.gf_dim*2, name='g_g3_conv'),name='g_bn_g3',is_training=is_training)
            # # g3_conv is (96 x 96 x self.gf_dim*2)
            # psi3 = lrelu(x3 + g3_conv)
            # psi3_conv = BN(conv2d_1(psi3, 1, name='g_p3_conv'),name='g_bn_p3',is_training=is_training)
            # # psi3_conv is (96 x 96 x 1)
            # psi3_conv = tf.nn.sigmoid(psi3_conv)
            # e3_ = e3_*psi3_conv
            """
            up_conv_block2
            """
            self.d3, self.d3_w, self.d3_b = deconv2d_my(tf.nn.relu(self.d2_1),k_size=3,d_size=1, name='g_d3', with_w=True)
            d3 = tf.nn.dropout(BN(self.d3,name='g_bn_d3',is_training=is_training), dropout_rate)
            d3 = tf.concat([d3, e3_], 3)
            # d3 is (96 x 96 x self.gf_dim*4*2)
            """
            conv_block2
            """
            self.d3_0 = BN(conv2d_(lrelu(d3), self.gf_dim*4, name='g_d3_0_conv_'),name='g_bn_d3_0',is_training=is_training)
            self.d3_1 = BN(conv2d_(lrelu(self.d3_0), self.gf_dim*4, name='g_d3_1_conv'),name='g_bn_d3_1',is_training=is_training)
            # self.d3_1 is (96 x 96 x self.gf_dim*4)
            self.d3_out=conv2d_(self.d3_1,1,k_h=1,k_w=1,name='g_d3_out_conv')
            """
            attention gate3
            """
            # g4 = tf.image.resize_images(lrelu(self.d3_1),[192,192],method=0)
            # # g4 is (192 x 192 x self.gf_dim*2)
            # x4 = BN(conv2d_1(lrelu(e2_), self.gf_dim, name='g_x4_conv'),name='g_bn_x4',is_training=is_training)
            # # x4_ is (192 x 192 x self.gf_dim)
            # g4_conv = BN(conv2d_1(g4, self.gf_dim, name='g_g4_conv'),name='g_bn_g4',is_training=is_training)
            # # g4_conv is (192 x 192 x self.gf_dim)
            # psi4 = lrelu(x4 + g4_conv)
            # psi4_conv = BN(conv2d_1(psi4, 1, name='g_p4_conv'),name='g_bn_p4',is_training=is_training)
            # # psi4_conv is (192 x 192 x 1)
            # psi4_conv = tf.nn.sigmoid(psi4_conv)
            # e2_ = e2_*psi4_conv
            """
            up_conv_block3
            """
            self.d4, self.d4_w, self.d4_b = deconv2d_my(tf.nn.relu(self.d3_1),k_size=3,d_size=1, name='g_d4', with_w=True)
            d4 = BN(self.d4,name='g_bn_d4',is_training=is_training)
            d4 = tf.concat([d4, e2_], 3)
            # d4 is (192 x 192 x self.gf_dim*2*2)
            """
            conv_block3
            """
            self.d4_0 = BN(conv2d_(lrelu(d4), self.gf_dim*2, name='g_d4_0_conv_'),name='g_bn_d4_0',is_training=is_training)
            self.d4_1 = BN(conv2d_(lrelu(self.d4_0), self.gf_dim*2, name='g_d4_1_conv'),name='g_bn_d4_1',is_training=is_training)
            # self.d4_1 is (192 x 192 x self.gf_dim*2)
            self.d4_out=conv2d_(self.d4_1,1,k_h=1,k_w=1,name='g_d4_out_conv')
            
            """
            attention gate4
            """
            # g5 = tf.image.resize_images(lrelu(self.d4_1),[384,384],method=0)
            # # g5 is (384 x384 x self.gf_dim)
            # x5 = BN(conv2d_1(lrelu(e1_), self.gf_dim/2, name='g_x5_conv'),name='g_bn_x5',is_training=is_training)
            # # x5_ is (384 x384 x self.gf_dim/2)
            # g5_conv = BN(conv2d_1(g5, self.gf_dim/2, name='g_g5_conv'),name='g_bn_g5',is_training=is_training)
            # # g5_conv is (384 x384 x self.gf_dim/2)
            # psi5 = lrelu(x5 + g5_conv)
            # psi5_conv = BN(conv2d_1(psi5, 1, name='g_p5_conv'),name='g_bn_p5',is_training=is_training)
            # # psi5_conv is (384 x384 x 1)
            # psi5_conv = tf.nn.sigmoid(psi5_conv)
            # e1_ = e1_*psi5_conv
            """
            up_conv_block4
            """
            self.d5, self.d5_w, self.d5_b = deconv2d_my(tf.nn.relu(self.d4_1),k_size=3,d_size=1,name='g_d5', with_w=True)
            d5 =BN(self.d5,name='g_bn_d5',is_training=is_training)
            d5 = tf.concat([d5, e1_], 3)
            # d5 is (384 x384 × self.gf_dim*1*2 )
            self.d5_0 = BN(conv2d_(lrelu(d5), self.gf_dim, name='g_d5_0_conv_'),name='g_bn_d5_0',is_training=is_training)
            self.d5_1 = BN(conv2d_(lrelu(self.d5_0), self.gf_dim, name='g_d5_1_conv'),name='g_bn_d5_1',is_training=is_training)
            # self.d5_1 is (384 x384 x self.gf_dim)
            
            self.d6 = conv2d_(lrelu(self.d5_1), self.output_c_dim, name='g_d6_0_conv_')

            return tf.nn.tanh(self.d6)

    def sampler(self, image,is_training=True,dropout_rate=0.8, y=None):
        with tf.variable_scope("generator") as scope:
            print('sampler')
            scope.reuse_variables()
            s = self.output_size
            s2, s4, s8 = int(s/2), int(s/4), int(s/8)
            image=tf.reshape(image, tf.stack([-1, self.image_size,self.image_size, self.input_c_dim ]))
            # image is (384 x 384 x input_c_dim)
            e1_ = BN(conv2d_(image, self.gf_dim, name='g_e1_conv_'),name='g_bn_1_',is_training=is_training)
            # e1_ is (384 x 384 x self.gf_dim)
            
            e1 = BN(conv2d(e1_, self.gf_dim, name='g_e1_conv'),name='g_bn_1',is_training=is_training)
            # e1 is (192 x 192 x self.gf_dim)
            e2_ = BN(conv2d_(lrelu(e1),self.gf_dim,name='g_cgm_e2_'),name='g_bn_2_',is_training=is_training)
            # e2_ = BN(conv2d_(lrelu(e1), self.gf_dim*2, name='g_e2_conv_'),name='g_bn_2_',is_training=is_training)
            # e2_ is (192 x 192 x self.gf_dim*2)
            
            
            e2 = BN(conv2d(lrelu(e2_), self.gf_dim*2, name='g_e2_conv'),name='g_bn_2',is_training=is_training)
            # e2 is (96 x 96 x self.gf_dim*2)
            e3_ = BN(conv2d_(lrelu(e2),2*self.gf_dim,name='g_cgm_e3_'),name='g_bn_3_',is_training=is_training)
            # e3_ = BN(conv2d_(lrelu(e2), self.gf_dim*4, name='g_e3_conv_'),name='g_bn_3_',is_training=is_training)
            # e3_ is (96 x 96 x self.gf_dim*4)
            
            e3 = BN(conv2d(lrelu(e3_), self.gf_dim*4, name='g_e3_conv'),name='g_bn_3',is_training=is_training)
            # e3 is (48 x 48 x self.gf_dim*4)
            e4_ = BN(conv2d_(lrelu(e3),4*self.gf_dim,name='g_cgm_e4_'),name='g_bn_4_',is_training=is_training)
            # e4_ = BN(conv2d_(lrelu(e3), self.gf_dim*8, name='g_e4_conv_'),name='g_bn_4_',is_training=is_training)
            # e4_ is (48 x 48 x self.gf_dim*8)
            
            e4 = BN(conv2d(lrelu(e4_), self.gf_dim*8, name='g_e4_conv'),name='g_bn_4',is_training=is_training)
            # e4 is (24 x 24 x self.gf_dim*8)
            e5_ = BN(conv2d_(lrelu(e4),8*self.gf_dim,name='g_cgm_e5_'),name='g_bn_5_',is_training=is_training)
            # e5_ = BN(conv2d_(lrelu(e4), self.gf_dim*16, name='g_e5_conv_'),name='g_bn_5_',is_training=is_training)
            # e5_ is (24 x 24 x self.gf_dim*16)

            """
            attention gate1
            """
            # g2 = tf.image.resize_images(lrelu(e5_),[48,48],method=0)
            # # g2 is (48 x 48 x self.gf_dim*16)
            # x2 = BN(conv2d_1(lrelu(e4_), self.gf_dim*4, name='g_x2_conv'),name='g_bn_x2',is_training=is_training)
            # # x2_ is (48 x 48 x self.gf_dim*4)
            # g2_conv = BN(conv2d_1(g2, self.gf_dim*4, name='g_g2_conv'),name='g_bn_g2_',is_training=is_training)
            # # g2_conv is (48 x 48 x self.gf_dim*4)
            # psi2 = lrelu(x2 + g2_conv)
            # psi2_conv = BN(conv2d_1(psi2, 1, name='g_p2_conv'),name='g_bn_p2',is_training=is_training)
            # # psi2_conv is (48 x 48 x 1)
            # psi2_conv = tf.nn.sigmoid(psi2_conv)
            # e4_ = e4_*psi2_conv
            """
            up_conv_block1
            """
            self.d2, self.d2_w, self.d2_b = deconv2d_my(tf.nn.relu(e5_),k_size=3,d_size=1,name='g_d2', with_w=True)
            d2 = tf.nn.dropout(BN(self.d2,name='g_bn_d2',is_training=is_training), dropout_rate)
            d2 = tf.concat([d2, e4_], 3)
            # d2 is (48 x 48 x self.gf_dim*16)
            """
            conv_block1
            """
            self.d2_0 = BN(conv2d_(lrelu(d2), self.gf_dim*8, name='g_d2_0_conv_'),name='g_bn_d2_0',is_training=is_training)
            self.d2_1 = BN(conv2d_(lrelu(self.d2_0), self.gf_dim*8, name='g_d2_1_conv'),name='g_bn_d2_1',is_training=is_training)
            # self.d2_1 is (48 x 48 x self.gf_dim*8)
            self.d2_out=conv2d_(self.d2_1,1,k_h=1,k_w=1,name='g_d2_out_conv')
            #self.d2_out is 48*48*1
            """
            attention gate2
            """
            # g3 = tf.image.resize_images(lrelu(self.d2_1),[96,96],method=0)
            # # g3 is (96 x 96 x self.gf_dim*4)
            # x3 = BN(conv2d_1(lrelu(e3_), self.gf_dim*2, name='g_x3_conv'),name='g_bn_x3',is_training=is_training)
            # # x3_ is (96 x 96 x self.gf_dim*2)
            # g3_conv = BN(conv2d_1(g3, self.gf_dim*2, name='g_g3_conv'),name='g_bn_g3',is_training=is_training)
            # # g3_conv is (96 x 96 x self.gf_dim*2)
            # psi3 = lrelu(x3 + g3_conv)
            # psi3_conv = BN(conv2d_1(psi3, 1, name='g_p3_conv'),name='g_bn_p3',is_training=is_training)
            # # psi3_conv is (96 x 96 x 1)
            # psi3_conv = tf.nn.sigmoid(psi3_conv)
            # e3_ = e3_*psi3_conv
            """
            up_conv_block2
            """
            self.d3, self.d3_w, self.d3_b = deconv2d_my(tf.nn.relu(self.d2_1),k_size=3,d_size=1, name='g_d3', with_w=True)
            d3 = tf.nn.dropout(BN(self.d3,name='g_bn_d3',is_training=is_training), dropout_rate)
            d3 = tf.concat([d3, e3_], 3)
            # d3 is (96 x 96 x self.gf_dim*4*2)
            """
            conv_block2
            """
            self.d3_0 = BN(conv2d_(lrelu(d3), self.gf_dim*4, name='g_d3_0_conv_'),name='g_bn_d3_0',is_training=is_training)
            self.d3_1 = BN(conv2d_(lrelu(self.d3_0), self.gf_dim*4, name='g_d3_1_conv'),name='g_bn_d3_1',is_training=is_training)
            # self.d3_1 is (96 x 96 x self.gf_dim*4)
            self.d3_out=conv2d_(self.d3_1,1,k_h=1,k_w=1,name='g_d3_out_conv')
            """
            attention gate3
            """
            # g4 = tf.image.resize_images(lrelu(self.d3_1),[192,192],method=0)
            # # g4 is (192 x 192 x self.gf_dim*2)
            # x4 = BN(conv2d_1(lrelu(e2_), self.gf_dim, name='g_x4_conv'),name='g_bn_x4',is_training=is_training)
            # # x4_ is (192 x 192 x self.gf_dim)
            # g4_conv = BN(conv2d_1(g4, self.gf_dim, name='g_g4_conv'),name='g_bn_g4',is_training=is_training)
            # # g4_conv is (192 x 192 x self.gf_dim)
            # psi4 = lrelu(x4 + g4_conv)
            # psi4_conv = BN(conv2d_1(psi4, 1, name='g_p4_conv'),name='g_bn_p4',is_training=is_training)
            # # psi4_conv is (192 x 192 x 1)
            # psi4_conv = tf.nn.sigmoid(psi4_conv)
            # e2_ = e2_*psi4_conv
            """
            up_conv_block3
            """
            self.d4, self.d4_w, self.d4_b = deconv2d_my(tf.nn.relu(self.d3_1),k_size=3,d_size=1, name='g_d4', with_w=True)
            d4 = BN(self.d4,name='g_bn_d4',is_training=is_training)
            d4 = tf.concat([d4, e2_], 3)
            # d4 is (192 x 192 x self.gf_dim*2*2)
            """
            conv_block3
            """
            self.d4_0 = BN(conv2d_(lrelu(d4), self.gf_dim*2, name='g_d4_0_conv_'),name='g_bn_d4_0',is_training=is_training)
            self.d4_1 = BN(conv2d_(lrelu(self.d4_0), self.gf_dim*2, name='g_d4_1_conv'),name='g_bn_d4_1',is_training=is_training)
            # self.d4_1 is (192 x 192 x self.gf_dim*2)
            self.d4_out=conv2d_(self.d4_1,1,k_h=1,k_w=1,name='g_d4_out_conv')
            
            """
            attention gate4
            """
            # g5 = tf.image.resize_images(lrelu(self.d4_1),[384,384],method=0)
            # # g5 is (384 x384 x self.gf_dim)
            # x5 = BN(conv2d_1(lrelu(e1_), self.gf_dim/2, name='g_x5_conv'),name='g_bn_x5',is_training=is_training)
            # # x5_ is (384 x384 x self.gf_dim/2)
            # g5_conv = BN(conv2d_1(g5, self.gf_dim/2, name='g_g5_conv'),name='g_bn_g5',is_training=is_training)
            # # g5_conv is (384 x384 x self.gf_dim/2)
            # psi5 = lrelu(x5 + g5_conv)
            # psi5_conv = BN(conv2d_1(psi5, 1, name='g_p5_conv'),name='g_bn_p5',is_training=is_training)
            # # psi5_conv is (384 x384 x 1)
            # psi5_conv = tf.nn.sigmoid(psi5_conv)
            # e1_ = e1_*psi5_conv
            """
            up_conv_block4
            """
            self.d5, self.d5_w, self.d5_b = deconv2d_my(tf.nn.relu(self.d4_1),k_size=3,d_size=1,name='g_d5', with_w=True)
            d5 =BN(self.d5,name='g_bn_d5',is_training=is_training)
            d5 = tf.concat([d5, e1_], 3)
            # d5 is (384 x384 × self.gf_dim*1*2 )
            self.d5_0 = BN(conv2d_(lrelu(d5), self.gf_dim, name='g_d5_0_conv_'),name='g_bn_d5_0',is_training=is_training)
            self.d5_1 = BN(conv2d_(lrelu(self.d5_0), self.gf_dim, name='g_d5_1_conv'),name='g_bn_d5_1',is_training=is_training)
            # self.d5_1 is (384 x384 x self.gf_dim)
            
            self.d6 = conv2d_(lrelu(self.d5_1), self.output_c_dim, name='g_d6_0_conv_')

            return tf.nn.tanh(self.d6)

    def save(self, checkpoint_dir, step):
        model_name = "pix2pix.model"
        model_dir = "%s_%s_%s_lr0.01" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
   
        model_dir = "%s_%s_%s_lr0.01" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False


        model_dir = "%s_%s_%s_lr0.01" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt_name='pix2pix.model-16'
        
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        
        return True



    def test(self,args):
        data = glob('E:/zys/GAN/datasets/{}/test_gen/*.mat'.format(self.dataset_name))
        #np.random.shuffle(data)
        batch_idxs = len(data) // self.batch_size
        
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        
        for idx in range(0, batch_idxs):
            batch_files = data[idx*self.batch_size:(idx+1)*self.batch_size]
            batch = [load_data(batch_file) for batch_file in batch_files]
            if (self.is_grayscale):
                batch_images = np.array(batch).astype(np.float32)
            else:
                batch_images = np.array(batch).astype(np.float32)   
#            print(batch_images.shape)
            samples = self.sess.run(
                self.fake_B_sample,
                feed_dict={self.real_data: batch_images}
            )
#            sample_A = batch_images[:, :, :,:self.input_c_dim]
            save_images(samples, batch_images,[self.batch_size, 4],
                       './{}/test_{:d}.bmp'.format(args.test_dir, idx))
#            save_images(sample_A, batch_images,[self.batch_size, 1],
#                        './{}/testTruth_{:d}.bmp'.format(args.test_dir, idx))
    def  finaltest(self,args):
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        
        for root,dirs,files in os.walk('E:/zys/GAN/datasets/{}/test_gen'.format(self.dataset_name)):
            for dir in dirs:
                dir_path=os.path.join(root,dir)
                imgs=os.listdir(dir_path)
                imgs.sort(key=lambda x:int(x[:-4]))
                i=0
                for img in imgs:
                    i=i+1
                    img_path=os.path.join(dir_path,img)
                    image = load_data(img_path)
                    shape = image.shape
                    image1 = load_data(img_path)
                    image2 = load_data(img_path)
                    image3 = load_data(img_path)
                    image = image.reshape((1,shape[0],shape[1],shape[2]))
                    image1 = image.reshape((1,shape[0],shape[1],shape[2]))
                    image2 = image.reshape((1,shape[0],shape[1],shape[2]))
                    image3 = image.reshape((1,shape[0],shape[1],shape[2]))
                    img_AB = np.append(image, image1, 0)
                    img_AB = np.append(img_AB, image2, 0)
                    img_AB = np.append(img_AB, image3, 0)
                    if (self.is_grayscale):
                        image = np.array(image).astype(np.float32)[:, :, :, None]
                    else:
                        img_AB = np.array(img_AB).astype(np.float32)   
#                       print(img_AB.shape)
                    
                    result = self.sess.run( self.fake_B_sample,feed_dict={self.real_data: img_AB})
                    result_shape = result.shape
                    #if i==0:
                     #   result_one_person =  result[0,:,:,:] 
                      #  result_one_person = np.reshape(result_one_person,[1,result_shape[1],result_shape[2],result_shape[3]])
                    #else :
                     #   im_one = np.reshape(result[0,:,:,:], [1,result_shape[1],result_shape[2],result_shape[3]])
                      #  result_one_person =  np.concatenate((result_one_person, im_one ), axis=0)  
                    save_dirs=args.finaltest_save_dirs

                    save_dir=os.path.join(save_dirs,dir)
                    word_name=os.path.exists(save_dir)
                    if not word_name:
                        os.makedirs(save_dir)
                    save_img_path = save_dir+'/'+str(i)+'.bmp'
                    save_images(result, img_AB,[self.batch_size, 9],save_img_path)

    def  finaltest3(self,args):
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        data=[]
        for root,dirs,files in os.walk('E:/zys/GAN/datasets/{}/test'.format(self.dataset_name),topdown='False'):
            for dir in dirs:
                dirs_path=os.path.join(root,dir)
                imgs=os.listdir(dirs_path)
                for img in imgs:
                    img_path=dirs_path+'/'+img
                    data.append(img_path)

        for root,dirs,files in os.walk('E:/zys/GAN/datasets/{}/test'.format(self.dataset_name)):
            for dir in dirs:
                dir_path=os.path.join(root,dir)
                imgs=os.listdir(dir_path)
                imgs.sort(key=lambda x:int(x[:-4]))
                i=0
                for img in imgs:
                    print(i)
                    i=i+1
                    img_path=os.path.join(dir_path,img)
                    image = load_data(img_path)
                    shape = image.shape
                    image = image.reshape((1,shape[0],shape[1],shape[2]))
                    batch_files = np.random.choice(data,self.batch_size-1)
                    # batch_files = data[num*self.batch_size:(num+1)*self.batch_size]
                    batch = [load_data(batch_file) for batch_file in batch_files]
                    img_AB=np.append(image,batch,0)
                    if (self.is_grayscale):
                        batch_images = np.array(img_AB).astype(np.float32)[:, :, :, None]

                    else:
                        batch_images = np.array(img_AB).astype(np.float32)
                    

                    result = self.sess.run( self.fake_B_sample,feed_dict={self.real_data: img_AB,
                                                                          self.is_training:False,
                                                                          self.dropout_rate:1.0})
                    result_shape = result.shape
                    #if i==0:
                     #   result_one_person =  result[0,:,:,:] 
                      #  result_one_person = np.reshape(result_one_person,[1,result_shape[1],result_shape[2],result_shape[3]])
                    #else :
                     #   im_one = np.reshape(result[0,:,:,:], [1,result_shape[1],result_shape[2],result_shape[3]])
                      #  result_one_person =  np.concatenate((result_one_person, im_one ), axis=0)  
                    save_dirs=args.finaltest_save_dirs

                    save_dir=os.path.join(save_dirs,dir)
                    word_name=os.path.exists(save_dir)
                    if not word_name:
                        os.makedirs(save_dir)
                    save_img_path = save_dir+'/'+str(i)+'.bmp'
                    save_images(result, img_AB,[self.batch_size, 9],save_img_path)

    def  finaltest4(self,args):
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        data=[]
        for root,dirs,files in os.walk('E:/zys/GAN/datasets/{}/test'.format(self.dataset_name),topdown='False'):
            for dir in dirs:
                dirs_path=os.path.join(root,dir)
                imgs=os.listdir(dirs_path)
                for img in imgs:
                    img_path=dirs_path+'/'+img
                    data.append(img_path)

        for root,dirs,files in os.walk('E:/zys/GAN/datasets/{}/test'.format(self.dataset_name)):
            for dir in dirs:
                dir_path=os.path.join(root,dir)
                imgs=os.listdir(dir_path)
                imgs.sort(key=lambda x:int(x[:-4]))
                i=0
                for img in imgs:
                    i=i+1
                    img_path=os.path.join(dir_path,img)
                    image = load_data(img_path)
                    shape = image.shape
                    image = image.reshape((1,shape[0],shape[1],shape[2]))
                    image2=load_data('E:/zys/GAN/datasets/{}/test/changyuan/6.mat'.format(self.dataset_name))
                    image3=load_data('E:/zys/GAN/datasets/{}/test/caofenglian/12.mat'.format(self.dataset_name))
                    image4=load_data('E:/zys/GAN/datasets/{}/test/gaosi/8.mat'.format(self.dataset_name))
                    image2 = image2.reshape((1,shape[0],shape[1],shape[2]))
                    image3 = image3.reshape((1,shape[0],shape[1],shape[2]))
                    image4 = image4.reshape((1,shape[0],shape[1],shape[2]))
                    # batch_files = np.random.choice(data,self.batch_size-1)
                    # batch_files = data[num*self.batch_size:(num+1)*self.batch_size]
                    # batch = [load_data(batch_file) for batch_file in batch_files]
                    img_AB=np.append(image,image2,0)
                    img_AB=np.append(img_AB,image3,0)
                    img_AB=np.append(img_AB,image4,0)
                    if (self.is_grayscale):
                        batch_images = np.array(img_AB).astype(np.float32)[:, :, :, None]

                    else:
                        batch_images = np.array(img_AB).astype(np.float32)
                    

                    result = self.sess.run( self.fake_B_sample,feed_dict={self.real_data: img_AB,self.is_training:False})
                    result_shape = result.shape
                    #if i==0:
                     #   result_one_person =  result[0,:,:,:] 
                      #  result_one_person = np.reshape(result_one_person,[1,result_shape[1],result_shape[2],result_shape[3]])
                    #else :
                     #   im_one = np.reshape(result[0,:,:,:], [1,result_shape[1],result_shape[2],result_shape[3]])
                      #  result_one_person =  np.concatenate((result_one_person, im_one ), axis=0)  
                    save_dirs=args.finaltest_save_dirs

                    save_dir=os.path.join(save_dirs,dir)
                    word_name=os.path.exists(save_dir)
                    if not word_name:
                        os.makedirs(save_dir)
                    save_img_path = save_dir+'/'+str(i)+'.bmp'
                    save_images(result, img_AB,[self.batch_size, 9],save_img_path)



    def  finaltest2(self,args):
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        data=[]
        for root,dirs,files in os.walk('E:/zys/GAN/datasets/{}/test'.format(self.dataset_name),topdown='False'):
            for dir in dirs:
                dirs_path=os.path.join(root,dir)
                imgs=os.listdir(dirs_path)
                for img in imgs:
                    img_path=dirs_path+'/'+img
                    data.append(img_path)
        np.random.shuffle(data)
        for num in range(0,97):
            batch_files = data[num*1:(num+1)*1]
            # batch_files = data[num*self.batch_size:(num+1)*self.batch_size]
            batch = [load_data(batch_file) for batch_file in batch_files]
            if (self.is_grayscale):
                batch_images = np.array(batch).astype(np.float32)

            else:
                batch_images = np.array(batch).astype(np.float32)

            result = self.sess.run( self.fake_B_sample,feed_dict={self.real_data: batch_images})
            result_shape = result.shape      
            save_dirs=args.finaltest_save_dirs

            # save_dir=os.path.join(save_dirs,dir)
            # word_name=os.path.exists(save_dir)
            # if not word_name:
            #     os.makedirs(save_dir)
            save_img_path = save_dirs+'/'+str(num)+'.bmp'
            save_images(result, batch_images,[self.batch_size, 9],save_img_path)   









