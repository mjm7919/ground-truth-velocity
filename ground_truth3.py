import cv2
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.morphology import medial_axis, skeletonize
cv2.destroyAllWindows()



def click_and_crop(event,x,y,flags,param):
    global refPt #, cropping
    if event==cv2.EVENT_LBUTTONDOWN:
        refPt.append([(x,y)])
#        cropping=True
    elif event==cv2.EVENT_LBUTTONUP:
        refPt.append([(x,y)])
#        cropping=False
        

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
#cap1 = cv2.VideoCapture(os.path.join(path,'flow_crop'+add2x[big]))#_2x.avi'))
cap1 = cv2.VideoCapture(os.path.join(path,'Series2/Series002_Stablized.avi'))#+add2x[big]))#_2x.avi'))

n_frames = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap1.get(cv2.CAP_PROP_FPS))

cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
res, frame1 = cap1.read()
frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
h,w = frame1.shape[:2]
cap1.release()
frame1_prev = frame1.copy()
#cv2.imshow('1',frame1)

refPt=[]
cropping=False
clone = frame1.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)
 
# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", frame1)
	key = cv2.waitKey(1) & 0xFF
 
	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		frame1 = clone.copy()
 
	# if the 'c' key is pressed, break from the loop
	elif key == ord("c"):
		break
 
# if there are two reference points, then crop the region of interest
# from teh image and display it
frame1=np.dstack((frame1,frame1,frame1))
cv2.rectangle(frame1,refPt[0][0],refPt[1][0],(0,255,0),2)
cv2.imshow('frame1',frame1)
accept_roi=input('accept roi? y/n: ')
if accept_roi=='y':
    if len(refPt) == 2:
    	roi = clone[refPt[0][0][0]:refPt[1][0][0], refPt[0][0][1]:refPt[1][0][1]]
    	cv2.imshow("ROI", roi)
#%%
kernel = np.ones((3,3),np.uint8)
kernel2 = np.zeros((11,11),np.uint8)
kernel2[5,5] = 4
boxFilter = np.ones((11,11))/125
kernel2=kernel2-boxFilter
cv2.destroyAllWindows()
#path='C:/Users/Matthew/Documents/Image_Analysis/RBC_flow/Series'
#cap1 = cv2.VideoCapture(os.path.join(path,'flow_crop'+add2x[big]))#_2x.avi'))
cap1 = cv2.VideoCapture(os.path.join(path,'Series2/Series002_Stablized.avi'))#+add2x[big]))#_2x.avi'))
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
w2=w*2
h2=h

smear = 1
wrangle=11
thresh = 120
tol=0.015
n=[]
y=[]
cap_space=cv2.imread(os.path.join(path,'Series2/smoothed ratio analysis.png'),0)
h,w=cap_space.shape[:2]

for k in np.arange(0,n_frames,1):
    status1, frame1 = cap1.read()
    frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    frame1 = frame1[25:frame1.shape[0]-25, 25:frame1.shape[1]-25]
#    frame1=np.maximum(fgmask1, diff) # broadcasting

#    frame1 = cv2.filter2D(frame1,-1,kernel2)

    
    if k==2:
        a=sum(sum(frame1-frame1_prev))
    if wrangle ==1:
        for i in range(0,h):
            for j in range(0,w):
                if (frame1_prev[i,j]-frame1_prev[i,j]*tol)<= frame1[i,j] <(frame1_prev[i,j]+frame1_prev[i,j]*tol):
                    frame1[i,j]=frame1_prev[i,j]
#                    y.append('yes')
                else:
                    frame1[i,j]=frame1[i,j]
#                    n.append('no')
    if smear ==1:
        for i in range(0,h):
            for j in range(0,w):
                if frame1[i,j] <thresh:
                    frame1[i,j]=100
                
    fgmask1 = fgbg.apply(frame1)
    fgmask1_MED=cv2.medianBlur(fgmask1,3)
    diff= fgmask1+fgmask1_MED
    
    diff=np.maximum(fgmask1, diff) # broadcasting
    fgmask1medDIFF=cv2.medianBlur(diff,7)
    fgmask1medDIFF=cv2.medianBlur(diff,3)
    
    f1track = np.hstack((frame1,fgmask1,fgmask1_MED,diff,fgmask1medDIFF))
    f=np.dstack((f1track,f1track,f1track))
    cv2.imshow('frame',f)
    cv2.waitKey(20)
    time.sleep(.2)  # secs
    print(k)
    frame1_prev = frame1.copy()

cap1.release()
cv2.destroyAllWindows()
