import cv2
import tensorflow as tf
import matplotlib.image as mpimg
import numpy as np


def recognizing_characters(img_original):
    #read image
    #img_original = mpimg.imread(path)
    #add a channels
    if len(img_original.shape) < 3:
        img_original = np.expand_dims(img_original, -1)
    if len(img_original.shape) == 3 and img_original.shape[2] == 3:
        img_new = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    else:
        img_new = img_original.copy()
    #to delete upper white rows
    list_white_rows =[]
    i = 0
    while np.all(img_new[i,:] == np.ones(img_new.shape[1:3], dtype=np.uint8)*255) == True:
        list_white_rows.append(i)
        if np.all(img_new[i,:] == np.ones(img_new.shape[1:3], dtype=np.uint8)*255) == False:
            break
        i += 1
    img_resized_tmp = np.delete(img_new,list_white_rows[:-5],axis=0)
    #to delete lower white rows
    list_white_rows_lower =[]
    i = 1
    while np.all(img_resized_tmp[-i,:] == np.ones(img_resized_tmp.shape[1:3], dtype=np.uint8)*255) == True:
        list_white_rows_lower.append(img_resized_tmp.shape[0]-i)
        if np.all(img_resized_tmp[-i,:] == np.ones(img_resized_tmp.shape[1:3], dtype=np.uint8)*255) == False:
            break
        i += 1


    img_resized = np.delete(img_resized_tmp,list_white_rows_lower[:-5],axis=0)

    if len(img_resized.shape) < 3:
        img_resized = np.expand_dims(img_resized, -1)
    else:
        img_resized = img_resized.copy()

    #create a white image with same dimensions
    white_image = np.ones(img_resized.shape, dtype=np.uint8)*255
    #transform to binary --> 90 for now but it can be changed
    ret, thresh = cv2.threshold(img_resized,90,255,cv2.THRESH_BINARY)
    #find contours, we can use none or simple as chain approx
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    h_list = hierarchy[0].tolist()
    list_contours = [[contours[i]] for i in range(len(contours))]
    list_index_delete = [i for i in range(len(contours)) if h_list[i][-1] != 0]
    for i in range(len(contours)):
        if h_list[i][3] != 0 and h_list[i][3] != -1:
            list_contours[h_list[i][3]].append(contours[i])
        else:
            pass
    for i in sorted(list_index_delete, reverse=True):
        list_contours.pop(i)
    #sort contours
    list_tuples_min = [(list_contours[i],list_contours[i][0][:,0][:,0].min()) for i in range(len(list_contours))]
    list_tuples_min.sort(key=lambda x: x[1])
    #extract the contours only
    new_contours = [list_tuples_min[i][0] for i in range(len(list_tuples_min))]
    #add info about character position in the equation
    full_image_h, full_image_w, full_image_channels = img_resized.shape
    tier1 = int(full_image_h/3)
    tier2 = int(full_image_h/3)*2
    tier3 = int(full_image_h/3)*3
    tier4 = int(full_image_w/3)
    tier5 = int(full_image_w/3)*2
    tier6 = int(full_image_w/3)*3
    min_x = [new_contours[c][0][:,0][:,0].min() for c in range(len(new_contours))]
    min_y = [new_contours[c][0][:,0][:,1].min() for c in range(len(new_contours))]
    max_x = [new_contours[c][0][:,0][:,0].max() for c in range(len(new_contours))]
    max_y = [new_contours[c][0][:,0][:,1].max() for c in range(len(new_contours))]
    middle_x = [int((new_contours[c][0][:,0][:,0][np.where(new_contours[c][0][:,0][:,1] == min_y[c])[0][0]] + max_x[c])/2) for c in range(len(new_contours))]
    position = []
    for i in range(len(min_x)):
        x_tmp,y_tmp,w_tmp,h_tmp = cv2.boundingRect(new_contours[i][0]) #i instead of 0 in the loop
        contour_drawing_tmp = cv2.drawContours(white_image.copy(), new_contours[i], -1, (0,0,0),-1)
        if (min_y[i] < int(1.3*tier1) and \
        new_contours[i][0][:,0][:,1][np.where(new_contours[i][0][:,0][:,0] == max_x[i])[0][0]] < int(1.3*tier1) and \
        new_contours[i][0][:,0][:,1][np.where(new_contours[i][0][:,0][:,0] == middle_x[i])[0][0]] < int(1.3*tier1)) and \
    abs(min_x[i]-new_contours[i][0][:,0][:,0][np.where(new_contours[i][0][:,0][:,1] == max_y[i])[0][0]]) < tier4 and \
    all([np.all(contour_drawing_tmp[j][middle_x[i]] == np.ones(img_resized.shape[2], dtype=np.uint8)*255) == \
 True for j in range(min_y[i]+10,max_y[i])]) and \
    all([np.all(contour_drawing_tmp[j][max_x[i]] == np.ones(img_resized.shape[2], dtype=np.uint8)*255) == \
 True for j in range(new_contours[i][0][:,0][:,1][np.where(new_contours[i][0][:,0][:,0] == max_x[i])[0][0]]+10,max_y[i])]) and \
    all([np.all(contour_drawing_tmp[max_y[i]][j] == np.ones(img_resized.shape[2], dtype=np.uint8)*255) == \
 True for j in range(new_contours[i][0][:,0][:,0][np.where(new_contours[i][0][:,0][:,1] == max_y[i])[0][0]]+10,max_x[i])]) and \
    len(new_contours[i]) == 1:
            position.append('root')
        elif 'root' in position and min_x[i] > new_contours[len(position) - position[::-1].index('root') - 1][0][:,0][:,0][np.where(new_contours[len(position) - position[::-1].index('root') - 1][0][:,0][:,1] == \
                                                                                                                                    max_y[len(position) - position[::-1].index('root') - 1])[0][0]] and max_x[i] < max_x[len(position) - position[::-1].index('root') - 1] and \
        min_y[i] < tier1 and max_y[i] < int(1.5*tier1):
            position.append('radicand-exp')
        elif 'root' in position and min_x[i] > new_contours[len(position) - position[::-1].index('root') - 1][0][:,0][:,0][np.where(new_contours[len(position) - position[::-1].index('root') - 1][0][:,0][:,1] == \
                                                                                                                                    max_y[len(position) - position[::-1].index('root') - 1])[0][0]] and max_x[i] < max_x[len(position) - position[::-1].index('root') - 1] and \
        max_y[i] > tier2 and min_y[i] > int(1.5*tier1):
            position.append('radicand-index')
        elif 'root' in position and min_x[i] > new_contours[len(position) - position[::-1].index('root') - 1][0][:,0][:,0][np.where(new_contours[len(position) - position[::-1].index('root') - 1][0][:,0][:,1] == max_y[len(position) - position[::-1].index('root') - 1])[0][0]] and max_x[i] < max_x[len(position) - position[::-1].index('root') - 1]:
            position.append('radicand')
        elif min_y[i] < tier1 and max_y[i] < int(1.5*tier1):
            position.append('exponent')
        elif max_y[i] > tier2 and min_y[i] > int(1.5*tier1):
            position.append('index')
        else:
            position.append('regular')

    list_images = []
    for c in range(len(new_contours)):
        x,y,w,h = cv2.boundingRect(new_contours[c][0])
        contour_drawing = cv2.drawContours(white_image.copy(), new_contours[c], -1, (0,0,0), -1)
        img = contour_drawing[int(0.95*y):int(1.1*(y+h)),int(0.95*x):int(1.1*(x+w))]
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
