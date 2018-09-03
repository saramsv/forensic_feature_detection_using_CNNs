##This script is used when one side of the images reduce to 128 already and now we want to 
##crop a 128 by 128 section on the image while making sure that the selection includes the tag
## And also the tags info needs to be updated which is done in this script
import csv
import os.path
from PIL import Image

def extract_coordinates(row, width, height):
    x = row[3][row[3].find('x":') + len('x":') : row[3].find(',"y')]
    y = row[3][row[3].find('"y":') + len('"y":') : row[3].find(',"width')]
    w = row[3][row[3].find('width":') + len('width":') : row[3].find(',"height')]
    h = row[3][row[3].find('height":') + len('height":') : row[3].find('}')]
    #print('iniinfo: '+ str(x)+' '+ str(y)+ ' ' +str(w)+ ' '+str(h))
    x = int(float(x)*width)
    y = int(float(y)*height)
    w = int(float(w)*width)
    h = int(float(h)*height)
    return x, y, w, h


def update_coordinates(row, x, y, w, h):
    row[3] = row[3].replace(row[3][row[3].find('x":') + len('x":')  : row[3].find(',"y')], str(x))
    row[3] = row[3].replace(row[3][row[3].find('"y":') + len('"y":') : row[3].find(',"width')], str(y))
    row[3] = row[3].replace(row[3][row[3].find('width":') + len('width":')  : row[3].find(',"height')], str(w))
    row[3] = row[3].replace(row[3][row[3].find('height":') + len('height":')  : row[3].find('}')], str(h))
    return row

'''
def resize_tags(row, img):
    im = Image.open(img)
    width, height = im.size
    x, y, w, h = extract_coordinates(row)
    if w > BASE_WIDTH and w > h:

    new_height =  
    new_width = 

    return img, row
'''


def crop_img(row, width, height, img):#width and height of the whole image
    x, y, w, h = extract_coordinates(row, width, height)
    ##Make the tags a bit biger that what's been reported

    if x-2 > 0:
        x = x - 2 ## take the starting point 3 pixels upper
        w = w + 5
    else:
        w = w + 3
    if y-2 > 0:
        y = y - 2
        h = h + 5
    else:
        h = h + 3
    #print('tag info = x, y, w, h is: '+ str(x) + '\t' + str(y) + '\t' + str(w) + '\t' + str(h))
    x_delta = 0
    y_delta = 0
    x_new = x
    y_new = y

    #print('x+128 = ' + str(x +BASE_WIDTH) + ' and the image wisth is: '+ str(width))    
    #print('y + 128 = ' + str(y +BASE_HEIGHT) + ' and the image wisth is: '+ str(height))    
    if x + BASE_WIDTH > width:
        x_delta = x + BASE_WIDTH - width
        x_new = x - x_delta
        #print("x is out as much as = " + str(x_delta))
        #print('so the x_new should be '+ str(x_new))
    if y + BASE_HEIGHT > height:
        y_delta = y + BASE_HEIGHT -height
        y_new = y - y_delta
        #print("y is out as much as = " + str(y_delta))
        #print('so the y_new should be '+ str(y_new))
    x = float(x_delta) / BASE_WIDTH
    y = float(y_delta) / BASE_HEIGHT
    w = float(w) /  BASE_WIDTH 
    h = float(h) / BASE_HEIGHT
    #print('new percentages for x, y, w, h are: ', str(x) + '\t' + str(y)+ '\t' + str(w) + '\t' + str(h) + str(w*BASE_WIDTH) + '\t' + str(h*BASE_HEIGHT))
    row = update_coordinates(row, x, y, w, h)
    #print('new row is: '+ str(row))
    #print('crop from: ', str(x_new) + '\t' +  str(y_new) + '\t' + str((x_new + BASE_WIDTH)) + '\t' +  str((y_new + BASE_HEIGHT)))
    cropped_img = img.crop((x_new, y_new, (x_new + BASE_WIDTH), (y_new + BASE_HEIGHT)))

    #import bpython
    #bpython.embed(locals())
    #break    

    return cropped_img, row


if __name__ == '__main__':
    csv_file = '/home/mousavi/Mask_RCNN_Sara/few_imgs_and_tags/high_frec_tag_data/high_frec_single_word_tag2.csv'
    where_to_read = '/home/mousavi/Mask_RCNN_Sara/few_imgs_and_tags/all_images/all_imgs_resized_128by/' 
    cropped_path = '/home/mousavi/Mask_RCNN_Sara/few_imgs_and_tags/all_images/all_imgs_resized_128by128_2/'

    f = open(csv_file)
    data = csv.reader(f)
    BASE_WIDTH = 128
    BASE_HEIGHT = 128
    new_width = 0
    new_height = 0

    cropped_imgs_metedata = []
    cropped_imgs_metedata.append(['row_number', 'X_id', 'user', 'location', 'image', 'tag', 'created', 'X__v', 'image1'])
    counter = 0
    im_count = 0
    for row in data:
        counter += 1
        if counter <  2:
            continue 
        else:
            row_number, X_id, user, location, image, tag, created, X__v, image1 = row
            readimg = 0
            #if ('scavenging' in tag) or (tag == 'larvae') or ('fly' in tag): #"UT01-16D_02_16_2016 (16).JPG" in image1:
            #if tag =='adult fly' or tag == 'larvae': #"UT01-16D_02_16_2016 (16).JPG" in image1:
            #img = where_to_read + image[image.find("2"):]#.replace(".JPG",").JPG")
            name = image1[image1.find('Photos/')+len('Photos/'): image1.find('JPG')+len('JPG')]
            #print(name)
            img = where_to_read + name
            if os.path.isfile(img):
                print("boud")
                readimg = Image.open(img)
                width, height = readimg.size
                #print("w, h: ",width, height)
                x, y, w, h = extract_coordinates(row, width, height)

                #print("x, y, w, h: ", x, y, w, h)
                if w <= BASE_WIDTH and h <=BASE_HEIGHT:#Then we know that the tag isn't bigger than 1024*1024
                    '''
                    if width < BASE_WIDTH and width <= height:
                        width_percent = BASE_WIDTH / float(width)
                        new_height = int(float(height) * float(width_percent))
                        new_width = BASE_WIDTH
                        readimg = readimg.resize((new_width, new_height), PIL.Image.ANTIALIAS)
                    elif height < BASE_HEIGHT and height <= width:
                        height_percent = BASE_HEIGHT /float(height)
                        new_width = int(float(width) * float(height_percent))
                        new_height = BASE_HEIGHT
                        readimg = readimg.resize((new_width, new_height), PIL.Image.ANTIALIAS)
                    else:#size is big enough. thus: no changes
                        new_width = width
                        new_height = height
                    '''
                    new_width = width
                    new_height = height
                    #name = cropped_path + image1
                    name_res = cropped_path + str(im_count) + ".jpg"
                    cropped,line = crop_img(row, new_width, new_height, readimg)
                    line[8] = str(im_count) + ".jpg"
                    im_count += 1
                    cropped.save(name_res)
                    cropped_imgs_metedata.append(line)
    #Write the new info in a csv file                    
    
    print(im_count)
    with open ('/home/mousavi/Mask_RCNN_Sara/few_imgs_and_tags/high_frec_tag_data/high_frec_128by128_2.csv','w') as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(cropped_imgs_metedata)
   
