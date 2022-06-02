#code to recognize singular characters in equations
import cv2
import tensorflow as tf

def recognizing_characters(path):
    #read image
    img_original = cv2.imread(path,-1)
    #transform to grays
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    #transform to binary --> 90 for now but it can be changed
    ret, thresh = cv2.threshold(img_gray,90,255,cv2.THRESH_BINARY)
    #find contours, we can use none or simple as chain approx
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
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
    list_min = []
    list_max = []
    for c in range(len(new_contours)):
        max_y = max(new_contours[c][:,0][:,1])
        min_y = min(new_contours[c][:,0][:,1])
        list_min.append(min_y)
        list_max.append(max_y)
    position = []
    for i in range(len(list_min)):
        if list_min[i] < tier1 and list_max[i] < tier2:
            position.append('exponent')
        elif list_max[i] > tier2 and list_min[i] > tier1:
            position.append('index')
        else:
            position.append('regular')
    #create a list of images
    list_images = []
    for c in range(len(new_contours)):
        x,y,w,h = cv2.boundingRect(new_contours[c])
        img = img_original.copy()[y:y+h,x:x+w]
        list_images.append(img)
    #create variables for size of full image and mean lenght and compare with the images in the list
    big_image_len = len(img_original)
    mean_len = sum([len(list_images[i]) for i in range(len(list_images))])/len(list_images)
    deleted_index = []
    for image in list_images:
        if len(image) == big_image_len:
            deleted_index.append(list_images.index(image))
        elif len(image) < 0.2*mean_len: #20% but it can be changed
            deleted_index.append(list_images.index(image))
    for i in deleted_index:
        position.pop(deleted_index[i])
        list_images.pop(deleted_index[i])
    character_position = [(list_images[i],position[i]) for i in range(len(list_images))]
    return {'list_images':list_images,
            'images_with_position': character_position}


def new_measurements(array):
    resized = cv2.resize(array, (45,45), interpolation = cv2.INTER_AREA)
    resized_bw = tf.image.rgb_to_grayscale(resized)
    resized_3  =  tf.image.grayscale_to_rgb(resized_bw)
    return resized_3

def test_data(path):
    list_images = recognizing_characters(path)['list_images']
    list_images_resized = []
    for img in list_images:
        resized_img = new_measurements(img)
        list_images_resized.append(resized_img)
    return list_images_resized

def test_data_with_positions(path):
    images_with_position = recognizing_characters(path)['images_with_position']
    return images_with_position

#if __name__ == "__main__":
#test
