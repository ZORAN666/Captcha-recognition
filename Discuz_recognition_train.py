import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, random, cv2

class Discuz():
    def __init__(self):
        # set up GPU
        os.environ["CUDA_VISIBLE_DEVISES"] = "0"
        self.config = tf.ConfigProto(allow_soft_placement=True)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
        self.config.gpu_options.allow_growth = True

        # data direction
        self.data_path = './Discuz/'
        self.log_dir = './Tb'
        # data parameter
        self.width = 30
        self.heigth= 100
        self.max_steps = 1000000
        self.test_images, self.test_labels, self.train_images, self.train_labels = self.get_images()
        self.train_size = len(self.train_images)
        self.test_size = len(self.test_images)
        # pointer of  batch_size
        self.train_ptr = 0
        self.test_ptr = 0
        self.char_set_len = 63
        self.max_captcha_size = 4
        # Placeholder
        self.X = tf.placeholder(tf.float32,shape=[None, self.heigth*self.width])
        self.Y = tf.placeholder(tf.float32,shape=[None, self.char_set_len*self.max_captcha_size])
        self.keep_prob = tf.placeholder(tf.float32)

    def get_images(self, rate = 0.2):
        # Load images
        images = os.listdir(self.data_path)
        # random sequence
        random.shuffle(images)
        # data set parameter
        images_num = len(images)
        train_num = int(images_num/(1+rate))
        train_images = images[:train_num]
        train_labels = list(map(lambda x: x.split('.')[0], train_images))
        test_images = images[train_num:]
        test_labels = list(map(lambda x: x.split('.')[0], test_images))
        return test_images, test_labels, train_images, train_labels

    # transform text to vector
    def text2vec(self, text):
        if len(text) > 4:
            print("maximum 4 characters")

        vector = np.zeros(4*63,dtype=int)
        def char2pos(char):
            if char == '_':
                k = 62
                return k
            k = ord(char) - 48
            if k > 9:
                k = ord(char) - 55
                if k > 35:
                    k = ord(char) - 61
                    if k > 61:
                        raise ValueError('No Map')
            return k
        for i, char in enumerate(text):
            index = i*63 + char2pos(char)
            vector[index] = 1
        return  vector

    # transform vector into characters
    def vec2text(self, vec):
        char_position = vec.nonzero()[0]
        text = []
        for i,c in enumerate(char_position):
            char_at_position = i
            char_index = c % 63
            if char_index < 10:
                char_code = char_index + ord('0')
            elif char_index < 36:
                char_code = char_index -10 + ord('A')
            elif char_index < 62:
                char_code = char_index -36 + ord('a')
            elif char_index == 62:
                char_code = ord('_')
            else:
                raise ValueError('error')
            char = chr(char_code)
            text.append(char)
        text = "".join(text)
        return text

    def get_next_batch(self, train_flag = True, batch_size = 100):
        # get data training set
        if train_flag == True:
            if (batch_size + self.train_ptr) < self.train_size:
                trains = self.train_images[self.train_ptr:(self.train_ptr + batch_size)]
                labels = self.train_labels[self.train_ptr:(self.train_ptr + batch_size)]
                self.train_ptr += batch_size
            else:
                new_ptr = (self.train_ptr + batch_size) % self.train_size
                trains = self.train_images[self.train_ptr:]+self.train_images[:new_ptr]
                labels = self.train_labels[self.train_ptr:]+self.train_labels[:new_ptr]
                self.train_ptr = new_ptr

            batch_x = np.zeros([batch_size, self.heigth*self.width])
            batch_y = np.zeros([batch_size, self.max_captcha_size*self.char_set_len])

            for index, train in enumerate(trains):
                image = np.mean(cv2.imread(self.data_path + train), -1)
                batch_x[index,:] = image.flatten()/255 # normalized
            for index, label in enumerate(labels):
                batch_y[index,:] = self.text2vec(label)

        # get data from testing set
        else:
            if (batch_size + self.test_ptr) < self.test_size:
                tests = self.test_images[self.test_ptr:(self.test_ptr + batch_size)]
                labels = self.test_labels[self.test_ptr:(self.test_ptr + batch_size)]
                self.test_ptr += batch_size
            else:
                new_ptr = (self.test_ptr + batch_size) % self.test_size
                tests = self.test_images[self.test_ptr:] + self.test_images[:new_ptr]
                labels = self.test_labels[self.test_ptr:] + self.test_labels[:new_ptr]
                self.test_ptr = new_ptr

            batch_x = np.zeros([batch_size, self.heigth * self.width])
            batch_y = np.zeros([batch_size, self.max_captcha_size * self.char_set_len])
            for index, test in enumerate(tests):
                image = np.mean(cv2.imread(self.data_path + test), -1)
                batch_x[index,:] = image.flatten()/255
            for index, label in enumerate(labels):
                batch_y[index,:] = self.text2vec(label)

        return batch_x, batch_y

    # CNN model
    def captcha_cnn(self, w_alpha = 0.01,b_alpha = 0.1):
        x = tf.reshape(self.X, shape=[-1, self.heigth, self.width, 1])
        w_c1 = tf.Variable(w_alpha*tf.random_normal([3,3,1,32]))
        b_c1 = tf.Variable(b_alpha*tf.random_normal([32]))
        covn1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x,w_c1,strides=[1,1,1,1],padding='SAME'), b_c1))
        covn1 = tf.nn.max_pool(covn1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

        w_c2 = tf.Variable(w_alpha*tf.random_normal([3,3,32,64]))
        b_c2 = tf.Variable(b_alpha*tf.random_normal([64]))
        covn2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(covn1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
        covn2 = tf.nn.max_pool(covn2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

        w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
        b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
        covn3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(covn2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
        covn3 = tf.nn.max_pool(covn3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

        w_d = tf.Variable(w_alpha*tf.random_normal([3328, 1024]))
        b_d = tf.Variable(b_alpha*tf.random_normal([1024]))
        dense_layer = tf.reshape(covn3,[-1, w_d.get_shape().as_list()[0]])
        dense_layer = tf.nn.relu(tf.nn.bias_add(tf.matmul(dense_layer,w_d),b_d))
        dense_layer = tf.nn.dropout(dense_layer, self.keep_prob)

        w_out = tf.Variable(w_alpha*tf.random_normal([1024, self.max_captcha_size*self.char_set_len]))
        b_out = tf.Variable(b_alpha*tf.random_normal([self.max_captcha_size*self.char_set_len]))
        out = tf.nn.bias_add(tf.matmul(dense_layer, w_out),b_out)

        return out

    def train_captcha_cnn(self):
        output = self.captcha_cnn()
        # Loss function
        diff = tf.nn.sigmoid_cross_entropy_with_logits(logits= output, labels = self.Y)
        loss = tf.reduce_mean(diff)
        tf.summary.scalar('loss',loss)

        # AdamOptimizer minimum cross-entropy loss
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        #calculate accuracy
        y = tf.reshape(output, [-1, self.max_captcha_size, self.char_set_len])
        y_labels = tf.reshape(self.Y, [-1, self.max_captcha_size, self.char_set_len])
        correct_prediction = tf.equal(tf.argmax(y, 2), tf.arg_max(y_labels, 2))
        accuracy  = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        merged = tf.summary.merge_all()
        saver = tf.train.Saver()
        with tf.Session(config = self.config) as sess:
            # write into the  direction
           train_writer = tf.summary.FileWriter(logdir=self.log_dir + '/train',graph=sess.graph)
           test_writer = tf.summary.FileWriter(logdir=self.log_dir + '/test')
           sess.run(tf.global_variables_initializer())

           # go through number  of  steps
           for i in range(self.max_steps):
               # shuffle set every 500  steps
                if i % 499 == 0:
                    self.test_images, self.test_labels, self.train_images, self.train_labels = self.get_images()
                # every 10 times to test
                if i % 10 ==0:
                    batch_x_test, batch_y_test = self.get_next_batch(False, 100)
                    summary, acc = sess.run([merged, accuracy], feed_dict = {self.X: batch_x_test, self.Y: batch_y_test, self.keep_prob: 1})
                    print('step: %d, accuracy: %f' % (i+1, acc))
                    test_writer.add_summary(summary, i )

                    if acc > 0.95:
                        train_writer.close()
                        test_writer.close()
                        saver.save(sess, "captcha_recognition.model", global_step=i)
                        print('The final step is: %d' % i)

                else:
                    batch_x,  batch_y = self.get_next_batch(True,100)
                    loss_value, _ = sess.run([loss, optimizer], feed_dict = {self.X: batch_x,self.Y: batch_y, self.keep_prob: 1})
                    print('Step: %d,   loss:%f' % (i + 1, loss_value))
                    curve = sess.run(merged, feed_dict={self.X: batch_x_test, self.Y: batch_y_test, self.keep_prob: 1})
                    train_writer.add_summary(curve, i)

                train_writer.close()
                test_writer.close()
                saver.save(sess, "Tb/captcha_recognition.model", global_step=self.max_steps)

if __name__ =='__main__':
    dz = Discuz()
    dz.train_captcha_cnn()





























