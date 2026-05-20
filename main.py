import argparse
import os
import scipy.misc
import numpy as np 
import tensorflow as tf 


from se_skip_gate import se_skip_gate






parser = argparse.ArgumentParser( description='abc')
parser.add_argument('--dataset_name', dest='dataset_name', default='OAI', help='name of the dataset')#OAI_test用于测试
parser.add_argument('--epoch', dest='epoch', type=int, default = 80)
parser.add_argument('--batch_size', dest= 'batch_size', type=int, default=4, help='# images in batch')
                   

parser.add_argument('--input_nc', dest='input_nc', type=int, default=1, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=5, help='# of output image channels') 
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.9, help='momentum term of adam')

parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='E:\zys\GAN\GAN_results_yuceshi\checkpoint_OAI', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='E:/zys/GAN/GAN_results_yuceshi/gan_seg_384/trainsample_OAI_loss', help='trainsample are saved here')
parser.add_argument('--sample_dir_', dest='sample_dir_', default='E:/zys/GAN/GAN_results_yuceshi/gan_seg_384/sample_OAI_loss', help='sample are saved here')
parser.add_argument('--finaltest_save_dirs',dest='finaltest_save_dirs',default='E:/zys/GAN/GAN_results/gan_seg_384/test_result_OAI', help='test result are saved here')

parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=5, help='weight on L1 ter1m in objective')

parser.add_argument('--task', dest='task',default='segmentation', help='generate,segmentation')
parser.add_argument('--phase', dest='phase', default='finaltest3', help='train, test, finaltest,finaltest2,finaltest3')

parser.add_argument('--train_size', dest='train_size', type=int, default=680, help='# images used to train') #680
parser.add_argument('--load_size', dest='load_size', type=int, default=404, help='scale images to this size') #404
parser.add_argument('--fine_size', dest='fine_size', type=int, default=384, help='then crop to this size') #384
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')#64
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer') #64

args = parser.parse_args()
 
def main():
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if args.task == 'generate':

        if not os.path.exists(args.sample_dir):
                os.makedirs(args.sample_dir)


        if not os.path.exists(args.sample_dir_):
                os.makedirs(args.sample_dir_)

        if not os.path.exists(args.finaltest_save_dirs) :
                os.makedirs(args.finaltest_save_dirs) 

        with tf.Session() as sess:
            model = se_skip_gate(sess, image_size=args.fine_size, batch_size=args.batch_size,
                        output_size=args.fine_size,L1_lambda=args.L1_lambda,
                        input_c_dim=args.input_nc, output_c_dim=args.output_nc,dataset_name=args.dataset_name,
                        checkpoint_dir=args.checkpoint_dir, sample_dir=args.sample_dir
                        ) 
            if args.phase == 'train':
                model.train(args)
            if args.phase == 'finaltest':
                model.finaltest(args)
            if args.phase == 'finaltest3':
                if not os.path.exists(args.finaltest_save_dirs):
                    os.makedirs(args.finaltest_save_dirs)
                model.finaltest3(args) 
            if args.phase == 'finaltest4':
                if not os.path.exists(args.finaltest_save_dirs):
                    os.makedirs(args.finaltest_save_dirs) 
                model.finaltest4(args)



    if args.task == 'segmentation':  
        if not os.path.exists(args.sample_dir):
                os.makedirs(args.sample_dir)

        if not os.path.exists(args.sample_dir_):
                os.makedirs(args.sample_dir_)
        with tf.Session() as sess:
            model = se_skip_gate(sess, image_size=args.fine_size, batch_size=args.batch_size,
                        output_size=args.fine_size, L1_lambda=args.L1_lambda,
                        input_c_dim=args.input_nc, output_c_dim=args.output_nc,dataset_name=args.dataset_name,
                        checkpoint_dir=args.checkpoint_dir, sample_dir=args.sample_dir) 
            if args.phase == 'train':
                model.train(args)
            if  args.phase == 'finaltest':
                model.finaltest(args)
            if args.phase == 'finaltest2':
                model.finaltest2(args)
            if args.phase == 'finaltest3':
                model.finaltest3(args)
              

if __name__ == '__main__':
    main()
 