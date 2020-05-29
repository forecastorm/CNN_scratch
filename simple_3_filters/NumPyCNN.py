import numpy 
import sys
import cv2

def save_map(feature_map,filter_num,name):

    # Save Numpy array to csv
    csv_name = 'open_eye_17_' + name + '_filter_' + str(filter_num) + '.csv';
    numpy.savetxt(csv_name, feature_map, delimiter=',', fmt='%d')

    # Save conv_map as img
    jpg_name = 'open_eye_17_' + name + '_filter_' + str(filter_num) + '.jpg'
    cv2.imwrite(jpg_name,feature_map)

def conv(img,conv_filter):

    if (len(img.shape)!= len(conv_filter.shape)-1):
        print("Error: Number of dimensions in conv filter and image do not match.")
        exit()

    if( len(img.shape)>2 or len(conv_filter.shape)>3):
        if(img.shape[-1] != conv_filter.shape[-1]):
            print("Error: Number of channels in both image and filter must match.")
            sys.exit()

    if(conv_filter.shape[1]!=conv_filter.shape[2]):
        print('Error: Filter must be square matrix, I.e. number of rows and columns must match.')
        sys.exit()
    
    #if filter diemensions are odd.
    if(conv_filter.shape[1]%2==0):
        print('Error: Filter must have an even size. I.e. number of rows and columns must be even.')
        sys.exit()

    #An empty feature map to hold the output of convolving the filter(s) with the image. 
    #this looks like stride 1 
    #feature map shape by [407, 498, 2]
    #image shape by [409,500]
    feature_maps = numpy.zeros((img.shape[0] - conv_filter.shape[1] +1,
                               img.shape[1] - conv_filter.shape[1] + 1,
                               conv_filter.shape[0]))

    #iterate through 2 filters,call conv_ on image with 2 filters 
    for filter_num in range(conv_filter.shape[0]):

        print("Filter ", filter_num)

        curr_filter = conv_filter[filter_num,:]
    
        if(len(curr_filter.shape)>2):
            print("Greater than 2")
        else:
            conv_map = conv_(img,curr_filter)

        feature_maps[:,:,filter_num] = conv_map
        # save feature map as csv and jpg
        save_map(feature_maps[:,:,filter_num],filter_num,'feature_map')

    return feature_maps
    
def conv_(img,conv_filter):

    # size 3 filter
    filter_size = conv_filter.shape[1]

    #409, 500
    result = numpy.zeros(img.shape)

    print("Result shape: ")
    print(result.shape)

    #looping through the image to apply the convolution operation
    """
    Getting the current region to get multiplied with the filter.
    How to loop through the image and get the region based on
    the image and filter sizes is the most tricky part of the convolution
    """

    #r from 1 to 407
    for r in numpy.uint16(numpy.arange(filter_size/2.0,img.shape[0]-filter_size/2.0+1)):
         #2 --- 1 to 498
        for c in numpy.uint16(numpy.arange(filter_size/2.0,img.shape[1]-filter_size/2.0+1)):
            
            # 3 by 3 
            curr_region = img[r - numpy.uint16(numpy.floor(filter_size/2.0)):r + numpy.uint16(numpy.ceil(filter_size/2.0)),
                              c - numpy.uint16(numpy.floor(filter_size/2.0)):c + numpy.uint16(numpy.ceil(filter_size/2.0))]
           
            #product for the feature map
            #3 by 3 feature result matrix 
            curr_result = curr_region * conv_filter

            # summing all the multiplication result
            conv_sum = numpy.sum(curr_result)
            result[r,c] = conv_sum

    final_result = result[numpy.uint16(filter_size/2.0):result.shape[0]-numpy.uint16(filter_size/2.0),
                          numpy.uint16(filter_size/2.0):result.shape[1]-numpy.uint16(filter_size/2.0)]

    return final_result


def relu(feature_map):
    
    relu_out = numpy.zeros(feature_map.shape)

    # -1 gives last index 
    for map_num in range(feature_map.shape[-1]):
        for r in numpy.arange(0,feature_map.shape[0]):
            for c in numpy.arange(0,feature_map.shape[1]):
                relu_out[r,c,map_num] = numpy.max([feature_map[r,c,map_num],0])

        save_map(relu_out[:,:,map_num],map_num,'relu_map')

    # map_relu by[407, 498, 2]
    return relu_out

# def pooling(feature_map, size= 2, stride = 2):

#     #pool_out size by (204,249,2)
#     pool_out = numpy.zeros((numpy.uint16((feature_map.shape[0]-size+1)/stride+1),
#                             numpy.uint16((feature_map.shape[1]-size+1)/stride+1),
#                             feature_map.shape[-1]))


#     for map_num in range(feature_map.shape[-1]):
#         r2 = 0
#         for r in numpy.arrange(0,feature_map.shape[0]-size+1, stride):
#             c2 =0 
#             for c in numpy.arrange(0,feature_map.shape[1]-size+1, stride):
#                 # pool_out[r2,c2,map_num] = numpy.max[]
#                 return

