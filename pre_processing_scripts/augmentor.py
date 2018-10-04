#flip, hori, verti, zoom, rotate, combination of these
#python augmentor.py sorted.tags.csv new_csv /Users/sara/Downloads/forensic/all_imgs/
from PIL import Image
import csv
import sys
import json
import random
import cv2
import os

random.seed(10)

def flip_img(im_name, image, flip_flag, tag_coordinate, width, height):
    flipped = cv2.flip(image, flip_flag)
    x1, y1, x2, y2 = tag_coordinate
    w = x2 - x1
    h = y2 - y1 
    if flip_flag == 0:
        #flipped_img = cv2.rectangle(flipped,(int(x1), int(height - y1 -1 - h)), (int(x2), int(height - y2 - 1 + h)), (0, 0, 255), 3)
        x1 = int(x1)
        y1 = int(height - y1 -1 - h)
        x2 = int(x2)
        y2 = int(height - y2 - 1 + h)
        new_name = im_name + "fliped_y.JPG"
    elif flip_flag == 1:
        #flipped_img = cv2.rectangle(flipped,(int(width - x1 - 1 - w), int(y1)), (int(width - x2 - 1 + w), int(y2)), (0, 0, 255), 3)
        x1 = int(width - x1 - 1 - w)
        y1 = int(y1)
        x2 = int(width - x2 - 1 + w)
        y2 = int(y2)
        new_name = im_name + "fliped_x.JPG"
    
    elif flip_flag == -1:
        #flipped_img = cv2.rectangle(flipped,(int(width - x1 - 1 - w),  int(height - y1 -1 - h)), (int(width - x2 - 1 + w), int(height - y2 -1 + h)), (0, 0, 255), 3)
        x1 = int(width - x1 - w -1)
        y1 = int(height - y1 -1 - h)
        x2 = int(width - x2 - 1 + w)
        y2 = int(height - y2 - 1 + h)
        new_name = im_name + "fliped_xy.JPG"
    
    #cv2.imwrite(new_name, flipped_img)
    cv2.imwrite(new_name, flipped)

    #TODO: x, y, width and height must be divided first (percentage)
    # Done, must be tested
    loc = Coord(x1 / float(width), y1 / float(height), x2 / float(width), y2 / float(height))
    return update_row(row[:], new_name, loc)


def extract_coor_percentage(loc): #loc = row[3] = the column in csv that has the coordinate
    loc1 = json.loads(loc)
    loc = loc1[0]['geometry']
    location = []
    location.append(loc['x'])
    location.append(loc['y'])
    location.append(loc['width'])
    location.append(loc['height'])
    return location

# coord is in percentages
def update_row(row, new_name, coord):
    loc = row[3]
    loc1 = json.loads(loc)
    loc1[0]['geometry']['x'] = coord.x1
    loc1[0]['geometry']['y'] = coord.y1
    loc1[0]['geometry']['width'] = coord.width
    loc1[0]['geometry']['height'] = coord.height
    loc_str = json.dumps(loc1)
    row[3] = loc_str
    row[8] = new_name
    return row
    
'''
tag_loc: The location of the tag (in original boundaries) 
that we're going to check if it's in the cropped part
cropped_coor: The coordinate of the cropped image within the original boundaries
'''
#TODO some times a big portion of the tag is included. 
#so, it is better to check if intersection_of_tag_and_cropped/tag > a_threshold, consider
#it as included
  
def is_included(tag_loc, cropped):
    if tag_loc.x1 > cropped.x1 and tag_loc.y1 > cropped.y1 \
        and tag_loc.x2 < cropped.x2 and tag_loc.y2 < cropped.y2:
        return True
    return False 
        

class Coord():
    # Values are not in percentages
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1 

    def getInPercentages(self, width, height):
        output = Coord(self.x1 / width, self.y1 / height, \
            self.x2 / width, self.y2 / height)
        return output

    def isEqual(self, coord):
        if self.x1 == coord.x1 and \
            self.y1 == coord.y1 and \
            self.x2 == coord.x2 and \
            self.y2 == coord.y2:
            return True

        return False

# Returns values in percentage
def get_scaled_tag_coor(tag, img, width, height):
    scaled_tag_x1 = (tag.x1 - img.x1) * width / float(img.width)
    scaled_tag_y1 = (tag.y1 - img.y1) * height / float(img.height)
    scaled_tag_w = (tag.x2 - tag.x1) * width / float(img.width)
    scaled_tag_h = (tag.y2 - tag.y1) * height / float(img.height)
    scaled = Coord(scaled_tag_x1, scaled_tag_y1, \
        scaled_tag_x1 + scaled_tag_w, scaled_tag_y1 + scaled_tag_h)

    return scaled 

def scale_img(tags_info, image, scales, img_rows):
    height, width = image.shape[:2]
    #row_num = 0
    new_rows = []
    im_name = img_rows[0][8][:img_rows[0][8].find('JPG')]
    row_count = 0
    for img_row in img_rows:
        new_rows.append(img_row[:])

        #[0:4] #the last one is not included. that means this is indices 0, 1, 2, 3
        x1, y1, x2, y2, tag = tags_info[row_count]
        tag_coord = Coord(x1, y1, x2, y2)
        row_count += 1
        new_im_x1 = random.randint(0, x1)
        new_im_y1 = random.randint(0, y1)
        for scale in scales:
            temp_w = int(scale * width) 
            temp_h = int(scale * height) 

            new_img_coord = Coord(new_im_x1, new_im_y1, new_im_x1 + temp_w, \
                new_im_y1 + temp_h)

            #if the tags itself is bigger than this cut/scaled part of the image, ignore it
            if tag_coord.width > new_img_coord.width or \
                tag_coord.height > new_img_coord.height: 
                continue

            while new_img_coord.y2 < tag_coord.y2 or new_img_coord.x2 < tag_coord.x2 or \
                new_img_coord.x2 > width or new_img_coord.y2 > height:
                new_img_coord.x1 = random.randint(0, tag_coord.x1)
                new_img_coord.y1 = random.randint(0, tag_coord.y1)
                new_img_coord.y2 = new_img_coord.y1 + new_img_coord.height
                new_img_coord.x2 = new_img_coord.x1 + new_img_coord.width

            # new_im is coordinates in the original image (for the new crop)

            # Cropping
            scaled_im = image[new_img_coord.y1 : new_img_coord.y2, \
                new_img_coord.x1 : new_img_coord.x2]

            # Resizing
            scaled_im = cv2.resize(scaled_im, (width, height), interpolation = cv2.INTER_AREA)

            scaled_tag = get_scaled_tag_coor(tag_coord, new_img_coord, width, height)
            '''
            # Draw rectangles
            scaled_im = cv2.rectangle(scaled_im, (int(scaled_tag.x1), int(scaled_tag.y1)),\
                 (int(scaled_tag.x2), int(scaled_tag.y2)), (0, 0, 255), 3)
            '''
            new_name = im_name + tag + str(scale) + ".JPG"
            cv2.imwrite(new_name, scaled_im)

            row_to_add = update_row(img_row, new_name, \
                scaled_tag.getInPercentages(width, height))
            new_rows.append(row_to_add[:])

            for tag_loc in tags_info:
                included_tag_name = tag_loc[4]
                tag_loc = Coord(tag_loc[0], tag_loc[1], tag_loc[2], tag_loc[3])
                if not tag_loc.isEqual(tag_coord) and is_included(tag_loc, new_img_coord):
                    included_tag = get_scaled_tag_coor(tag_loc, \
                        new_img_coord, width, height)

                    temp_row = update_row(row_to_add[:], new_name, \
                        included_tag.getInPercentages(width, height))
    
                    temp_row[5] = included_tag_name 
                    new_rows.append(temp_row)

    return new_rows


def append_rows(rows):
    for i in range(len(rows)):
        new_rows.append(rows[i])


def get_coordinate(width, height,location):#this location is the percentage
    x1 = int(float(location[0])*width)
    y1 = int(float(location[1])*height)
    w = int(float(location[2])*width)
    x2 = x1 + w  
    h =  int(float(location[3])*height)
    y2 = y1 + h 
    return [x1, y1, x2, y2]


def img_tags_coor(width, height, lines):
    tags = []
    for l in lines:
        coor = []
        coor = get_coordinate(width, height, extract_coor_percentage(l[3]))
        coor.append(l[5])
        tags.append(coor)
    return tags #this is a list of lists [[x1, y1, x2, y2, tag1],[x1, y1, x2, y2, tag2]...]



#################################
if __name__ == '__main__':
    f = open(sys.argv[1])
    res_csv = sys.argv[2]
    imgs_dir = sys.argv[3]

    data = csv.reader(f)
    new_rows = []
    first_row = 1
    scales = [0.3, 0.5, 0.8]

    next_img = 0

    img_rows = []
    img_name = ""

    for row in data:
        if first_row == 1:
            first_row = 0
            continue

        else:
            name = row[8][ : row[8].find('JPG')]
            location = extract_coor_percentage(row[3])
            if os.path.isfile(imgs_dir + row[8]):
                image = cv2.imread(imgs_dir + row[8], cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
                height, width = image.shape[:2]
                coor = get_coordinate(width, height, location) # coor = [x1, y1, x2, y2]

                new_rows.append(flip_img(name, image, 1, coor, width, height))
                new_rows.append(flip_img(name, image, 0, coor, width, height))
                new_rows.append(flip_img(name , image, -1, coor, width, height))

                if img_name == name:
                    img_rows.append(row[:])
                elif img_name == "":
                    img_name = name
                    img_rows.append(row[:])
                    im_obj = image
                else:
                    next_img +=1
                    tags_info = img_tags_coor(im_obj.shape[1], im_obj.shape[0],img_rows)
                    draw = im_obj
                    '''
                    for i in tags_info:
                        cv2.rectangle(draw, (i[0],i[1]), (i[2],i[3]), (0,0,255),3)
                    '''
                    cv2.imwrite(img_name + "JPG", draw)
                    rows = scale_img(tags_info, im_obj, scales, img_rows)
                    append_rows(rows)
                    img_name = name
                    img_rows = []
                    img_rows.append(row[:])
                    im_obj = image
                    if next_img == 5:
                        break

    with open(res_csv, 'w') as output:
        writer = csv.writer(output, lineterminator = '\n')
        writer.writerows(new_rows)
