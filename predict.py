#coding=utf-8
import tensorflow as tf
import cv2
import os
import numpy as np
import RPi.GPIO as GPIO
import time
import os
os.system ('sudo modprobe bcm2835-v4l2')


GPIO.setmode(GPIO.BOARD)
GPIO.setup(5,GPIO.OUT,initial=0)
def bell():
	GPIO.output(5,1)
	time.sleep(0.1)
	GPIO.output(5,0)

# 加载模型
load_start_time=time.time()
saver = tf.train.import_meta_graph("logs/model/model.ckpt-300.meta")
sess = tf.Session()
saver.restore(sess, "logs/model/model.ckpt-300")
image_size=64
graph=tf.get_default_graph()
input_x=graph.get_tensor_by_name("input:0")
keep_prob=graph.get_tensor_by_name("keep_prob:0")
score=graph.get_tensor_by_name("fc3/fc3:0")
softmax=graph.get_tensor_by_name("softmax:0")
output=graph.get_tensor_by_name("output:0")
load_time=time.time()-load_start_time
print("load time="+str(load_time)+"s")


def pre_whiten(x):
	mean=np.mean(x)
	std=np.std(x)
	std_adj=np.maximum(std,1.0/np.sqrt(x.size))
	y=np.multiply(np.subtract(x,mean),1/std_adj)
	return y

flag=0
out_score=[0,0,0,0,0,0,0]


def predict(q):
	while 1:
		image= q.get()
		global softmax,sess,out_score,flag
		#image=cv2.imread(image)
		image = image[:, :, (2, 1, 0)]
		image=cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
		image=pre_whiten(image)
		image=np.reshape(image,[-1,image_size,image_size,3])
		out_score=sess.run(softmax,feed_dict={input_x:image,keep_prob:1})
		out_score= [int(x *100) for x in out_score[0]] 
		
		if out_score[2]>80 or  out_score[3]>80  or  out_score[4]>80 or  out_score[5]>80 or  out_score[6]>80:
			flag+=1
			print("NG")
			if flag>1:
				print "bell"
				bell()
		else:
			flag=0
			print("ok")
#~ #predict("data/wrong_92.jpeg",sess)
#~ #predict("data/right_1.jpeg",sess)

#~ def predict(q):
	#~ while 1:
		#~ print "thread"
		#~ print q.get()
	

from PIL import Image, ImageTk
import cv2
import threading
from Queue import Queue
q=Queue()

cap = cv2.VideoCapture(0)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,512)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT,512)
t_start = time.time()
fps = 0

t=threading.Thread(target=predict,args=(q,))
t.start()
print cap.isOpened()
while cap.isOpened():
	
	ret, frame = cap.read()
	frame=frame[100:400,100:400,:]
	fps = fps + 1
	sfps = fps / ( time.time() - t_start )
	cv2.putText(frame, "FPS : " + str( int( sfps ) ), ( 10, 10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 255, 255, 255 ), 2 )
	if out_score:
		cv2.putText(frame, "score: " + str(out_score), ( 20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 255, 255, 255), 1)

	if fps%10==0:
		score_start=time.time()
		q.put(frame)
		
		cv2.putText(frame, "score: " + str(out_score), ( 20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 255, 255, 255), 1)
		score_time=time.time()-score_start
		print("score_time="+str(int(score_time*1000))+"ms")
		
	cv2.imshow('frame',frame)
	cv2.waitKey(1)
	
