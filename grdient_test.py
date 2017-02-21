'''
@author: YUNGCHI LIU
Gradient():
    Read a jpg file, and try to do gradient
MAIN():
    Compute avg. and normalize.
'''

import cv2
import numpy as np
import matplotlib.image as mpimg
from os import listdir
from os.path import isfile, join


def Gradient( src, color):
    
    src = cv2.GaussianBlur(  src, (3,3), 0)
    ## Convert it to gray 
    src_gray = cv2.cvtColor( src, cv2.COLOR_BGR2GRAY )
       
    
    sobel_x = cv2.Sobel( src_gray,cv2.CV_16S,1,0,ksize=3)
    abs_grad_x = cv2.convertScaleAbs( sobel_x)
  
    sobel_y = cv2.Sobel( src_gray,cv2.CV_16S,0,1,ksize=3)
    abs_grad_y = cv2.convertScaleAbs( sobel_y)
    
    grad = abs_grad_x*.5 + abs_grad_y*.5
    
    return grad

    
    
    
def Main():
    num_im = 100
    
    
    folder = "C:/Users/ASUS/Desktop/sample_drive/cam_5"
    files = []
    for i in range(num_im):
        file = listdir(folder)[i]
        if isfile(join(folder, file)):
            files.append(file)
    
    print("start...")
    num = 0
    ave_im =  np.zeros((2032,2032), np.uint8)
    
    for file in files:
        num += 1
        if num%10 == 0 :
            print( "Read", num, "pics.." )
             
        im = mpimg.imread(join(folder, file))
        im = Gradient(im, 1)
        ave_im = ave_im + (im / num_im)
        del im
        
        
    ave_im = cv2.normalize( ave_im, ave_im, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    print( "Image output...")
    cv2.imwrite( "AVERAGE.jpg", ave_im)
#     img = cv2.imread('393408642.jpg',1)
#     img = Gradient( img, 1)
#     cv2.imwrite( "IMAGE.jpg", img)
    
    
    
    
if __name__ == '__main__': 
    Main()