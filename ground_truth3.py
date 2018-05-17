import cv2
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.morphology import medial_axis, skeletonize
cv2.destroyAllWindows()
def plot_color_image(image,title,colorscheme,file_ext,colorbar,path):
#    sizes=np.shape(image)
    my_dpi=96
    pix=700
    plt.figure(figsize=(pix/my_dpi, pix/my_dpi), dpi=my_dpi)
    plt.imshow(image,cmap=colorscheme)
    if colorbar == 'yes':
        plt.colorbar(fraction=0.046, pad=0.04)
#        plt.colorbar(orientation='horizontal')
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    print('saving %s.%s' % (title,file_ext,))
    plt.savefig('%s.%s' % (os.path.join(path,title),file_ext,),dpi=my_dpi, bbox_inches='tight')
    plt.show()

def plot_image(image,title,colorscheme,file_ext,colorbar,path):
#    sizes=np.shape(image)
    my_dpi=96
    pix=700
    plt.figure(figsize=(pix/my_dpi, pix/my_dpi), dpi=my_dpi)
    plt.imshow(image,cmap=colorscheme,vmin=0,vmax=255)
    if colorbar == 'yes':
        plt.colorbar(fraction=0.046, pad=0.04)
#        plt.colorbar(orientation='horizontal')
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    print('saving %s.%s' % (title,file_ext,))
    plt.savefig('%s.%s' % (os.path.join(path,title),file_ext,),dpi=my_dpi, bbox_inches='tight')
    plt.show()

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 	# 1 standard deviation above and below median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the canny detection edged image
	return edged

plt.close('all')
add2x='.avi','_2x.avi'
big=0
path='C:/Users/Matthew/Documents/Image_Analysis/RBC_flow/Series'
cap1 = cv2.VideoCapture(os.path.join(path,'flow_crop'+add2x[big]))#_2x.avi'))
cap2 = cv2.VideoCapture(os.path.join(path,'flow_jpeg_crop'+add2x[big]))

n_frames = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap1.get(cv2.CAP_PROP_FPS))

cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
res, frame1 = cap1.read()
frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
res, frame2 = cap2.read()
frame2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
h,w = frame1.shape[:2]
cap1.release()
cap2.release()
frame1_prev = frame1.copy()
frame2_prev = frame2.copy()
cv2.imshow('1',frame1)
cv2.imshow('2',frame2)
#%%
kernel = np.ones((3,3),np.uint8)
kernel2 = np.zeros((11,11),np.uint8)
kernel2[5,5] = 3
boxFilter = np.ones((11,11))/125
kernel2=kernel2-boxFilter
cv2.destroyAllWindows()
path='C:/Users/Matthew/Documents/Image_Analysis/RBC_flow/Series'
cap1 = cv2.VideoCapture(os.path.join(path,'flow_crop'+add2x[big]))#_2x.avi'))
cap2 = cv2.VideoCapture(os.path.join(path,'flow_jpeg_crop'+add2x[big]))
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
w2=w*2
h2=h
out =cv2.VideoWriter(os.path.join(path,'norm_3T120.avi'),cv2.VideoWriter_fourcc(*'MJPG'),20,(w2,h2))

smear = 1
wrangle=11
thresh = 120
tol=0.01
n=[]
y=[]
for k in np.arange(0,n_frames,1):
    status1, frame1 = cap1.read()
    frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    status2, frame2 = cap2.read()
    frame2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    
    if k==2:
        a=sum(sum(frame1-frame1_prev))
        b=sum(sum(frame2-frame2_prev))
    if wrangle ==1:
        for i in range(0,h):
            for j in range(0,w):
                if (frame1_prev[i,j]-frame1_prev[i,j]*tol)<= frame1[i,j] <(frame1_prev[i,j]+frame1_prev[i,j]*tol):
                    frame1[i,j]=frame1_prev[i,j]
                    y.append('yes')
                else:
                    frame1[i,j]=frame1[i,j]
                    n.append('no')
#                    frame2[i,j]=0
    if smear ==1:
        for i in range(0,h):
            for j in range(0,w):
                if frame1[i,j] <thresh:
                    frame1[i,j]=0
                    frame2[i,j]=0
#                else:
#                    frame1[i,j]=frame1[i,j]
##                    frame2[i,j]=0
                
    fgmask1 = fgbg.apply(frame1)
    frame1_2dFilter = cv2.filter2D(frame1,-1,kernel2)
    fgmask1_2dFilter = fgbg.apply(frame1_2dFilter)
#    fgmask = cv2.dilate(fgmask,kernel,iterations = 1)
#    fgmask = cv2.erode(fgmask,kernel,iterations = 1)
    fgmask1_2dFilter_med = cv2.medianBlur(fgmask1_2dFilter,3)
    
    fgmask2 = fgbg.apply(frame2)
    frame2_2dFilter = cv2.filter2D(frame2,-1,kernel2)
    fgmask2_2dFilter = fgbg.apply(frame2_2dFilter)


    f1track = np.hstack((frame1,fgmask1))#,frame1_2dFilter,fgmask1_2dFilter))
#    f2track = np.hstack((frame2,fgmask2,frame2_2dFilter,fgmask2_2dFilter))
#    f = np.vstack((f1track,f2track))
    f=np.dstack((f1track,f1track,f1track))
    cv2.imshow('frame',f)
    cv2.waitKey(20)
#    time.sleep(.05)  # secs
    out.write(f)
    print(k)
    frame1_prev = frame1.copy()
    frame2_prev = frame2.copy()

cap1.release()
cap2.release()
out.release()
cv2.destroyAllWindows()
##%%
#og=cv2.imread('original.png',0)
#cv2.imshow('og',og)
