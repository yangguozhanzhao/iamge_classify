# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import input_data
import CNN
import argparse

def main():
    
    data_path = os.path.join(FLAGS.buckets, "")
    logs_dir=os.path.join(FLAGS.checkpointDir, "")
    
    n_classes = 7
    image_size = 64  # resize the image, if the input image is too large, training will be very slow.
    batch_size = 64
    max_step = 300 # with current parameters, it is suggested to use MAX_STEP>10k
    learning_rate = 0.0001 # with current parameters, it is suggested to use learning rate<0.0001
    
    train_image_list,train_label_list,test_image_list,test_label_list = input_data.get_files(data_path)
    train_image_batch,train_label_batch = input_data.get_batch(train_image_list,train_label_list,image_size,image_size,batch_size)
    test_image_batch,test_label_batch = input_data.get_batch(test_image_list,test_label_list,image_size,image_size,batch_size)



    x = tf.placeholder(tf.float32, shape=[None,image_size,image_size,3],name="input")
    y = tf.placeholder(tf.int32, shape=[None,],name="labels")
    keep_prob=tf.placeholder(tf.float32,name="keep_prob")
    
    model = CNN.CNN(n_classes, dropout_keep_prob=keep_prob)
    logits = model.lenet_inference(x)
    loss = model.loss(logits, y)
    acc = model.evaluation(logits, y)
    train_op = model.optimize(learning_rate)
    
    
    
    summary_op = tf.summary.merge_all()
    #summary_op = tf.merge_all_summaries() 
    
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter (logs_dir+'train/',sess.graph)
        test_writer =tf.summary.FileWriter (logs_dir+'test/',sess.graph)
        

        saver=tf.train.Saver()
        sess.run(tf.global_variables_initializer())
       # sess.run(tf.initialize_all_variables())
        
        # start queue runner
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        try:
            for step in np.arange(max_step):
                if coord.should_stop():
                    break
                tra_images,tra_labels = sess.run([train_image_batch, train_label_batch])
                sess.run(train_op,feed_dict={x:tra_images, y:tra_labels,keep_prob:0.5})
                if step % 50==0:
                    tra_loss, tra_acc = sess.run([loss, acc],feed_dict={x:tra_images, y:tra_labels,keep_prob:1.0})
                    summary_str = sess.run(summary_op,feed_dict={x:tra_images, y:tra_labels,keep_prob:1.0})
                    train_writer.add_summary(summary_str, step)

                    test_images, test_labels = sess.run([test_image_batch, test_label_batch])
                    test_loss, test_acc = sess.run([loss, acc], feed_dict={x:test_images, y:test_labels,keep_prob:1.0})
                    summary_str = sess.run(summary_op, feed_dict={x:test_images, y:test_labels,keep_prob:1.0})
                    test_writer.add_summary(summary_str, step) 
                    print('Step %d, train_loss = %.2f, train_acc = %.2f,test_loss=%.2f,test_acc=%.2f' 
                    %(step, tra_loss, tra_acc*100.0,test_loss,test_acc*100))
            model_path=os.path.join(logs_dir+'model/', 'model.ckpt')
            saver.save(sess,model_path,global_step=max_step)
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop() 
        # stop queue runner
        coord.request_stop()
        coord.join(threads)
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    #获得buckets路径
    parser.add_argument('--buckets', type=str, default='./data/',
                        help='input data path')
    #获得checkpoint路径
    parser.add_argument('--checkpointDir', type=str, default='./logs/',
                        help='output model path')
    FLAGS, _ = parser.parse_known_args()
    main()
