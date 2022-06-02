#code to recognize singular characters in equations
import cv2

def recognizing_characters(path):
    #read image
    img_original = cv2.imread(path)
    #transform to grays
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    #transform to binary --> 90 for now but it can be changed
    ret, thresh = cv2.threshold(img_gray,90,255,cv2.THRESH_BINARY)
    #find contours, we can use none or simple as chain approx
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #sort contours
    list_tuples_min = [(contours[i],contours[i][:,0][:,0].min()) for i in range(len(contours))]
    list_tuples_min.sort(key=lambda x: x[1])
    new_contours = [list_tuples_min[i][0] for i in range(len(list_tuples_min))]
    #create a list of images
    list_images = []
    for c in range(len(new_contours)):
        x,y,w,h = cv2.boundingRect(new_contours[c])
        img = img_original.copy()[y:y+h,x:x+w]
        list_images.append(img)
    #create variables for size of full image and mean lenght and compare with the images in the list
    big_image_len = len(img_original)
    mean_len = sum([len(list_images[i]) for i in range(len(list_images))])/len(list_images)
    for image in list_images:
        if len(image) == big_image_len:
            list_images.remove(image)
        elif len(image) < 0.2*mean_len: #20% but it can be changed
            list_images.remove(image)
    return list_images


def new_measurements(array):
    resized = cv2.resize(array, (45,45), interpolation = cv2.INTER_AREA)
    #resized_channel = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    ret, img_resized = cv2.threshold(resized,190,255,cv2.THRESH_BINARY)
    return img_resized


def test_data(path):
    list_images = recognizing_characters(path)
    list_images_resized = []
    for img in list_images:
        resized_img = new_measurements(img)
        list_images_resized.append(resized_img)
    return list_images_resized

#if __name__ == "__main__":
