from ast import If
from django.shortcuts import render
from joblib import load
import tensorflow as tf
import numpy
import cv2
import keras
from PIL import Image, ImageChops
import matplotlib.pyplot as plt
import imutils
from os import listdir

from numpy import asarray

from django.http import HttpResponse

# model = load('./model/CNN_brain_tumor.joblib')
localhost_save_option = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")
model = keras.models.load_model('./model/CNN-VGG16_brain_tumor.h5', options=localhost_save_option)

# model = load('./model/Brain_tumor_cnn.joblib')

def Welcome(request):
    return render(request, 'index.html')

def Result(request):
    if request.method == 'POST' and request.FILES['image']:
        f = request.FILES['image']
        img = Image.open(f).convert('RGB')
        # numpydata = asarray(img)
        # return HttpResponse(numpydata)

        # np_img = numpy.array(img)
        # img_data = cv2.resize(np_img, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)

        X, y = load_data(img, (64, 64))
        data = model.predict(X)

        if(data[0] > 0.5):
            hasil = True
        else:
            hasil = False
        # return HttpResponse(img_data.shape)
        # return HttpResponse(hasil)

    return render(request, 'result.html', {'hasil' : hasil})

# fungsi trim background
def trim(im):
    bg = Image.new("RGB", im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im.convert("RGB"), bg)
    diff = ImageChops.add(diff, diff)
    bbox = diff.getbbox()
    # if bbox:
    return im.crop(bbox)

# Fungsi untuk crop gambar
def crop_mri(image, plot=False):
    # Ubah gambar kedalam bentuk grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    morph = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    morph = cv2.erode(morph, None, iterations=2)
    morph = cv2.dilate(morph, None, iterations=2)

    # Cari kontur dari gambar yang sudah dimorph, kemudian ambil yang terbesar
    cnts = cv2.findContours(morph.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # Set extreme point
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # Crop gambar dengan 4 extreme point yang sudah diset diatas (left, right, top, bottom)
    cropped = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    if plot:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.subplot(1, 2, 2)
        plt.imshow(cropped)
        plt.title("Cropped Image")
    
    return cropped

def load_data(dir_list, image_size):
    X = []
    y = []
    image_width, image_height = image_size

    # for directory in dir_list:
    #     for filename in listdir(directory):
            # image = cv2.imread(directory+'/'+filename)
    # im = Image.open(directory + '/' + filename).convert('RGB') 
    trimmed = trim(dir_list)
    image = numpy.array(trimmed)
    image = crop_mri(image, plot=False)
    image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)

    # Normalisasi gambar
    # image = image / 255
    X.append(image)
    y.append([0])

    # Jika direktori asal memiliki nama 'yes' maka akan diberi label 1,
    # Jika bukan maka diberi label 0
    # print(directory[-3:])
    # if(directory[-3:] == 'yes'):
    #     y.append([1])
    # else:
    #     y.append([0])

    X = numpy.array(X)
    y = numpy.array(y)

    print(f'Number of examples is: {len(X)}')
    print(f'X shape is: {X.shape}')
    print(f'y shape is: {y.shape}')

    return X, y