import skimage.data
import numpy
import matplotlib
import NumPyCNN as numpycnn
import cv2

def showImage(img):
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Reading the image
img = cv2.imread('open_eye_17.jpg',0)
print(img.shape)

#save gray scale image
cv2.imwrite('open_eye_17_gray.jpg',img)

#save gray scale csv
numpy.savetxt('open_eye_17.csv', img, delimiter=',', fmt='%d')

l1_filter = numpy.zeros((2,3,3))

# vertical feature detection 
l1_filter[0,:,:] = numpy.array( [  [[-1,0,1],
                                   [-1,0,1],
                                   [-1,0,1]]  ])

l1_filter[1,:,:] = numpy.array( [[[-1,-1,-1],
                                  [0,0,0],
                                  [1,1,1]]])
                                  
# this is the conv layer
#feature map shape by [407, 498, 2]
#image shape by [409,500]
l1_feature_map = numpycnn.conv(img,l1_filter)


# this is the ReLU layer 


# map_relu by[407, 498, 2]
l1_feature_map_relu = numpycnn.relu(l1_feature_map)


# this is the pooling layer 
# shape by [204,249,2]
l1_feature_map_relu_pool = numpycnn.pooling(l1_feature_map_relu,2,2)





