import cv2
import tensorflow as tf
import matplotlib.image as mpimg
import numpy as np


def recognizing_characters(path):
    #read image
    img_original = mpimg.imread(path)
    #add a channels
    if len(img_original.shape) < 3:
        img_original = np.expand_dims(img_original, -1)
    if len(img_original.shape) == 3 and img_original.shape[2] == 3:
        img_new = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    else:
        img_new = img_original.copy()
    #create a white image with same dimensions
    white_image = np.ones(img_new.shape, dtype=np.uint8)*255
    #transform to binary --> 90 for now but it can be changed
    ret, thresh = cv2.threshold(img_new,90,255,cv2.THRESH_BINARY)
    #find contours, we can use none or simple as chain approx
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    h_list = hierarchy[0].tolist()
    contours = list(contours)
    deleted_index = []
    for i in h_list:
        if i[-1] == -1:
            deleted_index.append(h_list.index(i))
    for d in sorted(deleted_index, reverse=True):
        contours.pop(d)
    #sort contours
    list_tuples_min = [(contours[i],contours[i][:,0][:,0].min()) for i in range(len(contours))]
    list_tuples_min.sort(key=lambda x: x[1])
    #extract the contours only
    new_contours = [list_tuples_min[i][0] for i in range(len(list_tuples_min))]
    #add info about character position in the equation
    full_image_h, full_image_w, full_image_channels = img_original.shape
    tier1 = int(full_image_h/3)
    tier2 = int(full_image_h/3)*2
    tier3 = int(full_image_h/3)*3
    tier4 = int(full_image_w/3)
    tier5 = int(full_image_w/3)*2
    tier6 = int(full_image_w/3)*3
    min_x = [new_contours[c][:,0][:,0].min() for c in range(len(new_contours))]
    min_y = [new_contours[c][:,0][:,1].min() for c in range(len(new_contours))]
    max_x = [new_contours[c][:,0][:,0].max() for c in range(len(new_contours))]
    max_y = [new_contours[c][:,0][:,1].max() for c in range(len(new_contours))]
    middle_x = [int((new_contours[c][:,0][:,0][np.where(new_contours[c][:,0][:,1] == min_y[c])[0][0]]+ max_x[c])/2) for c in range(len(new_contours))]
    position = []
    for i in range(len(min_x)):
        if min_y[i] < tier1 and max_y[i] < int(1.5*tier1):
            position.append('exponent')
        elif max_y[i] > tier2 and min_y[i] > int(1.5*tier1):
            position.append('index')
        elif (min_y[i] < int(1.3*tier1) and \
            new_contours[i][:,0][:,1][np.where(new_contours[i][:,0][:,0] == max_x[i])[0][0]] < int(1.3*tier1) and \
            new_contours[i][:,0][:,1][np.where(new_contours[i][:,0][:,0] == middle_x[i])[0][0]] < int(1.3*tier1)) and \
        abs(min_x[i]-new_contours[i][:,0][:,0][np.where(new_contours[i][:,0][:,1] == max_y[i])[0][0]]) < tier4:
            position.append('root')
        elif 'root' in position and min_x[i] > new_contours[position.index('root')][:,0][:,0][np.where(new_contours[position.index('root')][:,0][:,1] == max_y[position.index('root')])[0][0]] and max_x[i] < max_x[position.index('root')]:
            position.append('radicand')
        else:
            position.append('regular')
    list_images = []
    for c in range(len(new_contours)):
        x,y,w,h = cv2.boundingRect(new_contours[c])
        contour_drawing = cv2.drawContours(white_image.copy(), new_contours, c, (0,0,0), 10)
        img = contour_drawing[int(0.70*y):y+int(1.25*h),int(0.70*x):x+int(1.25*w)]
        list_images.append(img)
    character_position = [(list_images[i],position[i]) for i in range(len(list_images))]
    return {'list_images':list_images,
            'images_with_position': character_position}

def new_measurements(array):
    resized = cv2.resize(array, (45,45), interpolation = cv2.INTER_AREA)
    return resized

def add_channels(array):
    return np.expand_dims(array, -1)

def three_channels(array):
    if array.shape[2] == 3:
        resized_bw = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        resized_3  =  cv2.cvtColor(resized_bw,cv2.COLOR_GRAY2RGB)
    elif array.shape[2] == 1:
        resized_3  =  cv2.cvtColor(array,cv2.COLOR_GRAY2RGB)
    return resized_3

def test_data(path):
    list_images = recognizing_characters(path)['list_images']
    list_images_resized = [new_measurements(img) for img in list_images]
    new_list_images = []
    for img in list_images_resized:
        if len(img.shape) < 3:
            new_img = add_channels(img)
        else:
            new_img = img
        new_list_images.append(new_img)
    new_list_images_1 = []
    for image in new_list_images:
        reformatted_img = three_channels(image)
        new_list_images_1.append(reformatted_img)
    return new_list_images_1

def test_data_with_positions(path):
    images_with_position = recognizing_characters(path)['images_with_position']
    return images_with_position
