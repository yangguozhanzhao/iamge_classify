#coding=utf-8
import os
import tensorflow as tf
import matplotlib as mpl
mpl.use('tkagg')  
import matplotlib.pyplot as plt
import numpy as np
import cv2


# 数据处理
def get_files(file_dir):
	a = []
	label_a = []
	b = []
	label_b = []
	c = []
	label_c = []
	d= []
	label_d = []
	e = []
	label_e = []
	f = []
	label_f = []
	g = []
	label_g = []
	for file in os.listdir(file_dir):
		name = file.split('_')
		
		if name[0]=='a':
			a.append(file_dir + file)
			label_a.append(0)
		elif name[0]=="b":
			b.append(file_dir + file)
			label_b.append(1)
		elif name[0]=="c":
			c.append(file_dir + file)
			label_c.append(2)
		elif name[0]=="d":
			d.append(file_dir + file)
			label_d.append(3)
		elif name[0]=="e":
			e.append(file_dir + file)
			label_e.append(4)
		elif name[0]=="f":
			f.append(file_dir + file)
			label_f.append(5)
		else:
			g.append(file_dir + file)
			label_g.append(6)
	print('There are %d a\n %d b\n  %d c\n %d d\n  %d e\n %d f\n  %d g\n' %(len(a), len(b),len(c),len(d), len(e),len(f),len(g)))

	image_list = np.hstack((a, b, c,d,e,f,g))
	label_list = np.hstack((label_a, label_b,label_c,label_d, label_e,label_f,label_g))
	temp = np.array([image_list, label_list])
	temp = temp.transpose()
	np.random.shuffle(temp)

	image_list = list(temp[:, 0])
	label_list = list(temp[:, 1])
	label_list = [int(i) for i in label_list]

	# 分离一部分作为测试集
	number=int(len(image_list)*0.8)
	print("train_set length is:%d" % number)
	train_image_list=image_list[:number]
	val_image_list=image_list[number:]
	
	train_label_list=label_list[:number]
	val_label_list=label_list[number:]
	return train_image_list, train_label_list, val_image_list,val_label_list

def get_batch(image,label,image_W,image_H,batch_size):
	
	image = tf.cast(image,tf.string)
	label = tf.cast(label,tf.int32)
	#tf.cast()用来做类型转换

	input_queue = tf.train.slice_input_producer([image,label])
	# #加入队列

	label = input_queue[1]
	image_contents = tf.read_file(input_queue[0])

	#jpeg或者jpg格式都用decode_jpeg函数，其他格式可以去查看官方文档
	image = tf.image.decode_jpeg(image_contents,channels=3)
	image = tf.image.resize_image_with_crop_or_pad(image,300,300)
	
	image=tf.image.resize_images(image, (image_W, image_H), method=1)

	#对resize后的图片进行标准化处理
	image = tf.cast(image, tf.float32)
	# image = tf.image.random_saturation(image,0,5)
	#image=tf.image.random_contrast(image,0.1,0.6)
	#image=tf.image.random_hue(image,0.5)
	image = tf.image.per_image_standardization (image)

	image_batch,label_batch = tf.train.batch([image,label],batch_size = batch_size,num_threads=8,capacity = 3*batch_size)

	label_batch = tf.reshape(label_batch,[batch_size])
	return image_batch,label_batch
	#获取两个batch，两个batch即为传入神经网络的数据

if __name__ == '__main__':
	BATCH_SIZE = 4
	CAPACITY = 15
	IMG_W = 64
	IMG_H = 64

	train_dir = 'data/'

	train_image_list, train_label_list, val_image_list,val_label_list = get_files(train_dir)
	image_batch,label_batch = get_batch(train_image_list,train_label_list,IMG_W,IMG_H,BATCH_SIZE)

	with tf.Session() as sess:
		i=0
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord = coord)
		try:
			while not coord.should_stop() and i<2:
			#提取出两个batch的图片并可视化。
				img,label = sess.run([image_batch,label_batch])

				for j in np.arange(BATCH_SIZE):
					print('label: %d'%label[j])
					print(img[j,:,:,:].shape)
					plt.imshow(img[j,:,:,:])
					plt.show()
				i+=1
		except tf.errors.OutOfRangeError:
			print('done!')
		finally:
			coord.request_stop()
		coord.join(threads)
