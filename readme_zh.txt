main.py是参数配置以及启动文件
ops.py和utils.py是模块存放文件
environment.yml是环境配置文件
se_skip_gate.py是同时使用了se注意力、Attention gate以及多尺度机制的模型文件
zys_res2rgb_oai.m是将模型分割出的结果转换为单张彩色rgb图像的文件。
zys_rgb2tissue_oai.m是将彩色rgb图像中的每种组织单独提取出的文件。
zys_mat6_oai.m是数据集制作文件。
gan_chushi.py是原始GAN模型文件。
gan_msg.py是使用多尺度的模型文件。
gan_ag.py是使用注意力门控的模型文件。
gan_se.py是使用se注意力机制的文件。
gan_se_ag.py是使用se注意力机制和ag机制的文件。
