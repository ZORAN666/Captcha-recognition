from urllib.request import urlretrieve
import time, random, os

class Discuz():
    def __init__(self):
        # Discuz captcha generating address
        self.url = 'http://cuijiahua.com/tutrial/discuz/index.php?label='

    def random_captcha_text(self, captcha_size = 4):
        # generate captcha contents
        number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v', 'w', 'x', 'y', 'z']
        char_set = number + alphabet
        captcha_text = []
        for i in range(captcha_size):
            char = random.choice(char_set)
            captcha_text.append(char)
        captcha_text = ''.join(captcha_text)
        return captcha_text

    def download_discuz(self, nums = 60000):
        dirname = './Discuz'
        if dirname not in os.listdir():
            os.mkdir(dirname)
        for i in range(nums):
            label = self.random_captcha_text()
            print("picture %d : %s" %(i+1,label))
            urlretrieve(url=self.url + label, filename= dirname + '/' + label +'.jpg')
            time.sleep(0.5)
        print('Done.')

if __name__ == '__main__':
    dz = Discuz()
    dz.download_discuz()
