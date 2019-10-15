from PIL import Image
import os
import imageio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pic_dic = '/home/xie/Data/CGAN/results/CGAN_mnist_64_62'
gif_name = '/home/xie/Data/tmp/3.gif'
filenames = []

pics = sorted(os.listdir(pic_dic))

def get_gif(t=0.1):
    imgs = []
    for pic_name in pics:
        pic_path = os.path.join(pic_dic, pic_name)
        temp = Image.open(pic_path)
        imgs.append(temp)

    imgs[0].save(gif_name, save_all=True, append_images=imgs, duration=t)

get_gif()

