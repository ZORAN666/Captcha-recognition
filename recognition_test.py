import tensorflow as tf
import numpy as np
import Discuz_recognition_train


def test_captcha_recognition(captcha_image, captcha_label):
    output = dz.train_captcha_cnn()
    saver = tf.train.Saver()
    with tf.Session(config=dz.config) as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        for i in range(len(captcha_label)):
            image = captcha_image[i].flatten()
            label = captcha_label[i]
            prediction = tf.argmax(tf.reshape(output,[-1,dz.max_captcha_size, dz.char_set_len]),2)
            text_list = sess.run(prediction, feed_dict={dz.X: [image], dz.keep_prob:1})
            text = text_list[0].tolist()
            vector = np.zeros(dz.max_captcha_size*dz.char_set_len)
            i = 0
            for n in text:
                vector[i*dz.char_set_len + n] = 1
                i +=1
            prediction_text = dz.vec2text(vector)
            print("label:{}  prediction:{}".format(dz.vec2text(label), prediction_text))

if __name__ == '__main__':
    dz = Discuz_recognition_train.Discuz()
    batch_x = batch_y = dz.get_next_batch(False, 5)
    test_captcha_recognition(batch_x, batch_y)




