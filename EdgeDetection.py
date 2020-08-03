import os
from PIL import Image
from skimage.color import rgb2gray
import numpy as np
import cv2
import matplotlib.pyplot as plt
%matplotlib inline
from scipy import ndimage
from google.colab.patches import cv2_imshow

from google.colab import drive
drive.mount('/content/drive',force_remount=True)
#Variable used for total number of images
q=1
z=3

for j in range(q,z):
  i=str(j)
  #PLEASE SET PATH FOR THE TEST IMAGE AS PER SYSTEM . 
  #NOTE : PLEASE KEEP IMAGE FORMAT AS 'Final_Image_(Integer).jpg'
  if(j!=2):
    imagebeing_read = 'drive/My Drive/Images/Final_Image_'+i+'.jpg'
  else:
    imagebeing_read = 'drive/My Drive/Images/Final_Image_'+i+'.jpeg'
  oimg=cv2.imread(imagebeing_read)
  img=cv2.GaussianBlur(oimg,(5,5),0)


  #THIS IS PATH FOR WRITING THE IMAGE 
  # PLEASE EDIT ONLY THE PART BEFORE
  first_GaussianBlur = 'drive/My Drive/Edge_Detection_End/Image'+i+'/Image'+i+'_GaussianBlurred_1.jpg'
  cv2.imwrite(first_GaussianBlur,img)
  nimg=cv2.GaussianBlur(img,(3,3),0)
  second_GaussianBlur = 'drive/My Drive/Edge_Detection_End/Image'+i+'/Image'+i+'_GaussianBlurred_2.jpg'
  cv2.imwrite(second_GaussianBlur,nimg)
  
  
  v = np.mean(nimg)
  sigma = np.std(nimg)
  lowThresh = int(max(0, (1.0 - sigma) * 2*v))
  high_Thresh = int(min(255, (1.0 + sigma) * 2*v))
  edges_Canny = cv2.Canny(nimg,lowThresh,high_Thresh)

  #Path used for writing the Canny Image Detected.
  WriteImage = 'drive/My Drive/Edge_Detection_End/Image'+i+'/Image'+i+'_Canny.jpg'
  cv2.imwrite(WriteImage,edges_Canny)
  

for j in range(q,z):
  i=str(j)
  if(j!=2):
    #The image is again being read here. Again set path according to system
    imagebeing_read = 'drive/My Drive/Images/Final_Image_'+i+'.jpg'
  else:
    imagebeing_read = 'drive/My Drive/Images/Final_Image_'+i+'.jpeg'
  oimg=cv2.imread(imagebeing_read)
  img=cv2.GaussianBlur(oimg,(5,5),0)
  edges_Laplacian=cv2.Laplacian(img,cv2.CV_64F)
  #Laplacian Image is written here
  WriteImage_Laplacian = 'drive/My Drive/Edge_Detection_End/Image'+i+'/Image'+i+'_Laplacian.jpg'
  cv2.imwrite(WriteImage_Laplacian,edges_Laplacian)
  ReadImage_Laplacian=WriteImage_Laplacian
  edges_Laplacian=cv2.imread(ReadImage_Laplacian)
  img= cv2.cvtColor(edges_Laplacian, cv2.COLOR_BGR2GRAY)
  mean_value=np.mean(img)
  #Linear Contrast Enhancement 
  mean_value=mean_value*3.1
  #Thresholding performed to form definite boundaries
  ret,th1 = cv2.threshold(img,mean_value,255,cv2.THRESH_BINARY)
  th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
  th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
  #Creating Array for performing different type of thresholding
  titles = ['Original Image', 'Global Thresholding (v = 127)',
              'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
  images = [img, th1, th2, th3]

  th1=cv2.medianBlur(th1,3)
  WriteImage = 'drive/My Drive/Edge_Detection_End/Image'+i+'/Image'+i+'_Laplacian_GaussianFilter_5x5_Global.jpg'
  cv2.imwrite(WriteImage,th1)

for j in range(q,z):
  i=str(j)
  if(j!=2):
    imagebeing_read = 'drive/My Drive/Images/Final_Image_'+i+'.jpg'
  else:
    imagebeing_read = 'drive/My Drive/Images/Final_Image_'+i+'.jpeg'
  img=cv2.imread(imagebeing_read)
  #YUV TRANSFORM OF IMAGES BEING PERFORMED
  yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
  #YUV IMAGE BEING WRITTEN - Please Edit PATH ACCORDINGLY
  YUV_Write = 'drive/My Drive/Edge_Detection_End/Image'+i+'/Image'+i+'YUV.jpg'
  cv2.imwrite(YUV_Write,yuv)
  img = cv2.imread(YUV_Write)
  image=img
  #CONTRAST ENHANCEMENT
  new_image = np.zeros(image.shape, image.dtype)
  alpha=0.7
  beta=0
  for y in range(image.shape[0]):
      for x in range(image.shape[1]):
        for c in range(image.shape[2]):
              new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
  WriteImage_YUV = 'drive/My Drive/Edge_Detection_End/Image'+i+'/Image'+i+'_yuv_LinearTransformed.jpg'
  cv2.imwrite(WriteImage_YUV,new_image)

for j in range(q,z):
  i=str(j)
  imageRead = 'drive/My Drive/Edge_Detection_End/Image'+i+'/Image'+i+'_yuv_LinearTransformed.jpg'
  img = cv2.imread(imageRead,cv2.IMREAD_GRAYSCALE)
  img = cv2.GaussianBlur(img,(3,3),0)
  h = cv2.Sobel(img, cv2.CV_64F, 0, 1)
  v = cv2.Sobel(img, cv2.CV_64F, 1, 0)
  edges_Sobel = cv2.add(h, v)
  imageWrite = 'drive/My Drive/Edge_Detection_End/Image'+i+'/Image'+i+'_Sobel.jpg'
  cv2.imwrite(imageWrite,edges_Sobel)

for j in range(q,z):
  i=str(j)
  from PIL import Image
  import math
  path = "drive/My Drive/Edge_Detection_End/Image"+i+"/Image"+i+"YUV.jpg" # Your image path
  im = cv2.imread(path)
  width=im.shape[1]
  height=im.shape[0]
  img = Image.open(path)
  newimg = Image.new("RGB", (width, height), "white")
  for x in range(1, width-1):  # ignore the edge pixels for simplicity (1 to width-1)
      for y in range(1, height-1): # ignore edge pixels for simplicity (1 to height-1)

          # initialise Gx to 0 and Gy to 0 for every pixel
          Gx = 0
          Gy = 0

          # top left pixel
          p = img.getpixel((x-1, y-1))
          r = p[0]
          g = p[1]
          b = p[2]

          # intensity ranges from 0 to 765 (255 * 3)
          intensity = r + g + b

          # accumulate the value into Gx, and Gy
          Gx += -intensity
          Gy += -intensity

          # remaining left column
          p = img.getpixel((x-1, y))
          r = p[0]
          g = p[1]
          b = p[2]

          Gx += -2 * (r + g + b)

          p = img.getpixel((x-1, y+1))
          r = p[0]
          g = p[1]
          b = p[2]

          Gx += -(r + g + b)
          Gy += (r + g + b)

          # middle pixels
          p = img.getpixel((x, y-1))
          r = p[0]
          g = p[1]
          b = p[2]

          Gy += -2 * (r + g + b)

          p = img.getpixel((x, y+1))
          r = p[0]
          g = p[1]
          b = p[2]

          Gy += 2 * (r + g + b)

          # right column
          p = img.getpixel((x+1, y-1))
          r = p[0]
          g = p[1]
          b = p[2]

          Gx += (r + g + b)
          Gy += -(r + g + b)

          p = img.getpixel((x+1, y))
          r = p[0]
          g = p[1]
          b = p[2]

          Gx += 2 * (r + g + b)

          p = img.getpixel((x+1, y+1))
          r = p[0]
          g = p[1]
          b = p[2]

          Gx += (r + g + b)
          Gy += (r + g + b)

          # calculate the length of the gradient (Pythagorean theorem)
          length = math.sqrt((Gx * Gx) + (Gy * Gy))

          # normalise the length of gradient to the range 0 to 255
          length = length / 4328 * 255

          length = int(length)

          # draw the length in the edge image
          #newpixel = img.putpixel((length,length,length))
          newimg.putpixel((x,y),(length,length,length))
  writePath = 'drive/My Drive/Edge_Detection_End/Image'+i+'/Image'+i+'_Sobel_Manual.jpg'
  newimg.save(writePath)

for j in range(q,z):
  i=str(j)
  imageRead = 'drive/My Drive/Edge_Detection_End/Image'+i+'/Image'+i+'_Sobel_Manual.jpg'
  image=cv2.imread(imageRead)
  Gaussian = cv2.GaussianBlur(image,(5,5),0)
  Gaussian = cv2.GaussianBlur(Gaussian,(3,3),0)
  median=cv2.medianBlur(image,5)
  median=cv2.medianBlur(median,3)
  cv2.imwrite('drive/My Drive/Edge_Detection_End/Image'+i+'/Image'+i+'_Sobel_Manual_MedianFilter_5x5.jpg',median)
  cv2.imwrite('drive/My Drive/Edge_Detection_End/Image'+i+'/Image'+i+'_Sobel_Manual_GaussianFilter_5x5.jpg',Gaussian)
  img= cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
  mean_value=np.mean(img)
  mean_value=mean_value*3.1
  ret,th1 = cv2.threshold(img,mean_value,255,cv2.THRESH_BINARY)
  th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
  th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
  
  titles = ['Original Image', 'Global Thresholding (v = 127)',
              'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
  images = [img, th1, th2, th3]

  cv2.imwrite('drive/My Drive/Edge_Detection_End/Image'+i+'/Image'+i+'_Sobel_Manual_MedianFilter_5x5_Global.jpg',th1)
  cv2.imwrite('drive/My Drive/Edge_Detection_End/Image'+i+'/Image'+i+'_Sobel_Manual_MedianFilter_5x5_AdaptiveMeanThresholding.jpg',th2)
  cv2.imwrite('drive/My Drive/Edge_Detection_End/Image'+i+'/Image'+i+'_Sobel_Manual_MedianFilter_5x5_GaussianMeanThresholding.jpg',th3)
  img= cv2.cvtColor(Gaussian, cv2.COLOR_BGR2GRAY)
  mean_value=np.mean(img)
  mean_value=mean_value*3.1
  ret,th1 = cv2.threshold(img,mean_value,255,cv2.THRESH_BINARY)
  th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
  th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

  titles = ['Original Image', 'Global Thresholding (v = 127)',
              'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
  images = [img, th1, th2, th3]

  cv2.imwrite('drive/My Drive/Edge_Detection_End/Image'+i+'/Image'+i+'_Sobel_Manual_GaussianFilter_5x5_Global.jpg',th1)
  cv2.imwrite('drive/My Drive/Edge_Detection_End/Image'+i+'/Image'+i+'_Sobel_Manual_GaussianFilter_5x5_AdaptiveMeanThresholding.jpg',th2)
  cv2.imwrite('drive/My Drive/Edge_Detection_End/Image'+i+'/Image'+i+'_Sobel_Manual_GaussianFilter_5x5_GaussianMeanThresholding.jpg',th3)

  imageRead = 'drive/My Drive/Edge_Detection_End/Image'+i+'/Image'+i+'_Sobel.jpg'
  image=cv2.imread(imageRead)
  Gaussian = cv2.GaussianBlur(image,(5,5),0)
  Gaussian = cv2.GaussianBlur(Gaussian,(3,3),0)
  median=cv2.medianBlur(image,5)
  median=cv2.medianBlur(median,3)
  cv2.imwrite('drive/My Drive/Edge_Detection_End/Image'+i+'/Image'+i+'_Sobel_MedianFilter_5x5.jpg',median)
  cv2.imwrite('drive/My Drive/Edge_Detection_End/Image'+i+'/Image'+i+'_Sobel_GaussianFilter_5x5.jpg',Gaussian)
  img= cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
  mean_value=np.mean(img)
  mean_value=mean_value*3.1
  ret,th1 = cv2.threshold(img,mean_value,255,cv2.THRESH_BINARY)
  th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
  th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
  
  titles = ['Original Image', 'Global Thresholding (v = 127)',
              'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
  images = [img, th1, th2, th3]

  cv2.imwrite('drive/My Drive/Edge_Detection_End/Image'+i+'/Image'+i+'_Sobel_MedianFilter_5x5_Global.jpg',th1)
  cv2.imwrite('drive/My Drive/Edge_Detection_End/Image'+i+'/Image'+i+'_Sobel_MedianFilter_5x5_AdaptiveMeanThresholding.jpg',th2)
  cv2.imwrite('drive/My Drive/Edge_Detection_End/Image'+i+'/Image'+i+'_Sobel_MedianFilter_5x5_GaussianMeanThresholding.jpg',th3)
  img= cv2.cvtColor(Gaussian, cv2.COLOR_BGR2GRAY)
  mean_value=np.mean(img)
  mean_value=mean_value*3.1
  ret,th1 = cv2.threshold(img,mean_value,255,cv2.THRESH_BINARY)
  th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
  th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

  titles = ['Original Image', 'Global Thresholding (v = 127)',
              'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
  images = [img, th1, th2, th3]

  cv2.imwrite('drive/My Drive/Edge_Detection_End/Image'+i+'/Image'+i+'_Sobel_GaussianFilter_5x5_Global.jpg',th1)
  cv2.imwrite('drive/My Drive/Edge_Detection_End/Image'+i+'/Image'+i+'_Sobel_GaussianFilter_5x5_AdaptiveMeanThresholding.jpg',th2)
  cv2.imwrite('drive/My Drive/Edge_Detection_End/Image'+i+'/Image'+i+'_Sobel_GaussianFilter_5x5_GaussianMeanThresholding.jpg',th3)

for j in range (q,z):
  i=str(j)
  img_Canny = cv2.imread('drive/My Drive/Edge_Detection_End/Image'+i+'/Image'+i+'_Canny.jpg',cv2.IMREAD_GRAYSCALE)
  img_Laplacian = cv2.imread('drive/My Drive/Edge_Detection_End/Image'+i+'/Image'+i+'_Laplacian_GaussianFilter_5x5_Global.jpg',cv2.IMREAD_GRAYSCALE)
  img_Sobel = cv2.imread('drive/My Drive/Edge_Detection_End/Image'+i+'/Image'+i+'_Sobel_Manual_MedianFilter_5x5_GaussianMeanThresholding.jpg',cv2.IMREAD_GRAYSCALE)
  if j!=2:
    img_image = cv2.imread('drive/My Drive/Images/Final_Image_'+i+'.jpg')
  else:
    img_image = cv2.imread('drive/My Drive/Images/Final_Image_'+i+'.jpeg')
  width=img_image.shape[1]
  height=img_image.shape[0]
  for y in range(1,width-1):
    for x in range(1,height-1):
      ## Checking if all three images have pixel values as edge
      count=0
      count_Canny=False
      count_Laplacian=False
      count_Sobel=False
      if(img_Canny[x][y]>200):
        count=count+1
        count_Canny=True
      if(img_Laplacian[x][y]>200):
        count=count+1
        count_Laplacian=True
      if(img_Sobel[x][y]>200):
        count=count+1
        count_Sobel=True
      if count==3 :
        img_image[x][y][0]=255
        img_image[x][y][1]=255
        img_image[x][y][2]=255
      #Checking if the neighbouring pixels are marked edge pixels
      elif count==2:
        if(img_Canny[x-1][y]==255 and img_Canny[x+1][y]==255) or (img_Canny[x][y+1]==255 and img_Canny[x][y-1]==255) or (img_Canny[x-1][y+1]==255 and img_Canny[x+1][y-1]==255) or (img_Canny[x+1][y+1]==255 and img_Canny[x-1][y-1]==255):
            img_image[x][y][0]=255
            img_image[x][y][1]=255
            img_image[x][y][2]=255
        elif(count_Laplacian==False):
          if(img_Laplacian[x-1][y]==255 and img_Laplacian[x+1][y]==255) or (img_Laplacian[x][y+1]==255 and img_Laplacian[x][y-1]==255) or (img_Laplacian[x-1][y+1]==255 and img_Laplacian[x+1][y-1]==255) or (img_Laplacian[x+1][y+1]==255 and img_Laplacian[x-1][y-1]==255):
            img_image[x][y][0]=255
            img_image[x][y][1]=255
            img_image[x][y][2]=255
        elif(count_Sobel==False):
          if(img_Sobel[x-1][y]==255 and img_Sobel[x+1][y]==255) or (img_Sobel[x][y+1]==255 and img_Sobel[x][y-1]==255) or (img_Sobel[x-1][y+1]==255 and img_Sobel[x+1][y-1]==255) or (img_Sobel[x+1][y+1]==255 and img_Sobel[x-1][y-1]==255):
            img_image[x][y][0]=255
            img_image[x][y][1]=255
            img_image[x][y][2]=255
        else:
          img_image[x][y][0]=0
          img_image[x][y][1]=0
          img_image[x][y][2]=0
      else:
        img_image[x][y][0]=0
        img_image[x][y][1]=0
        img_image[x][y][2]=0
        
  writePath = 'drive/My Drive/Edge_Detection_End/Image'+i+'/Image'+i+'_EnsembledImage.jpg'
  cv2.imwrite(writePath,img_image)
  


for j in range(q,z):
  i=str(j)
  image = cv2.imread('drive/My Drive/Edge_Detection_End/Image'+i+'/Image'+i+'_EnsembledImage.jpg', 0)
  kernel_sharpening = np.array([[-1,-1,-1], 
                                  [-1, 9,-1],
                                  [-1,-1,-1]])
  sharpened = cv2.filter2D(image, -1, kernel_sharpening)
  ima=cv2.GaussianBlur(sharpened,(3,3),0)
  cv2.imwrite('drive/My Drive/Edge_Detection_End/Image'+i+'/Image'+i+'_EnsembledImage_Sharpened.jpg',ima)


