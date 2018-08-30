import tensorflow as tf
import numpy as np
import crack_captcha


def test_captcha_recognition(captcha_image, captcha_label):
    output = cracking.train_captcha_cnn()
    saver = tf.train.Saver
    with tf.Session(config=cracking.config) as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        for i in range(len(captcha_label)):
            image = captcha_image[i].flatten()
            label = captcha_label[i]
            prediction = tf.argmax(tf.reshape(output,[-1,cracking.max_captcha_size, cracking.char_set_len]),2)
            text_list = sess.run(prediction, feed_dict={cracking.X: [image], cracking.keep_prob:1})
            text = text_list[0].tolist()
            vector = np.zeros(cracking.max_captcha_size*cracking.char_set_len)
            i = 0
            for n in text:
                vector[i*cracking.char_set_len + n] = 1
                i +=1
            prediction_text = cracking.vec2text(vector)
            print("label:{}  prediction:{}".format(cracking.vec2text(label), prediction_text))

if __name__ == '__main__':
    cracking = crack_captcha.Cracker()
    batch_x = batch_y = cracking.get_next_batch(False, 5)
    test_captcha_recognition(batch_x, batch_y)
