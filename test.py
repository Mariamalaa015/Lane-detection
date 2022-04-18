import cv2
import matplotlib.pyplot as plt
import numpy as np

def abs_sobel_thresh(img, orient='x', thresh=(20, 100)):
    """
    #--------------------- 
    # This function applies Sobel x or y, and then 
    # takes an absolute value and applies a threshold.
    #
    """
    # Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))

    # Scale to 8-bit (0 - 255) then convert to type = np.uint8    
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
   
    # Create a binary mask where mag thresholds are met  
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 255

    # Return the result
    return binary_output


def get_combined_gradients(img, thresh_x, thresh_y, thresh_mag, thresh_dir):
    """
    #---------------------
    # This function isolates lane line pixels, by focusing on pixels
    # that are likely to be part of lane lines.
    # I am using Red Channel, since it detects white pixels very well. 
    #
    """
    rows, cols = img.shape[:2]

    R_channel = img[:,:, 2]

    sobelx = abs_sobel_thresh(R_channel, 'x', thresh_x)
    sobely = abs_sobel_thresh(R_channel, 'y', thresh_y)

    # combine sobelx, sobely
    gradient_combined = np.zeros_like(sobelx).astype(np.uint8)
    gradient_combined[((sobelx > 1)) | ((sobely > 1))] = 255  # | (R > 1)] = 255

    return gradient_combined


def channel_thresh(channel, thresh=(80, 255)):
    """
    #---------------------
    # This function takes in a channel of an image and
    # returns thresholded binary image
    # 
    """
    binary = np.zeros_like(channel)
    binary[(channel > thresh[0]) & (channel <= thresh[1])] = 255
    return binary


def get_combined_hls(img, th_h, th_l, th_s):
    """
    #---------------------
    # This function takes in an image, converts it to HLS colorspace, 
    # extracts individual channels, applies thresholding on them
    #
    """

    # convert to hls color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    rows, cols = img.shape[:2]
    
    H = hls[:,:, 0]
    L = hls[:,:, 1]
    S = hls[:,:, 2]

    h_channel = channel_thresh(H, th_h)
    l_channel = channel_thresh(L, th_l)
    s_channel = channel_thresh(S, th_s)
    
    # Trying to use Red channel, it works even better than S channel sometimes, 
    # but in cases where there is shadow on road and road color is different, 
    # S channel works better. 
    hls_comb = np.zeros_like(s_channel).astype(np.uint8)
    hls_comb[((s_channel > 1) & (l_channel == 0)) | ((s_channel == 0) & (l_channel > 1))] = 255 
    # trying to use both S channel and R channel
    #hls_comb[((s_channel > 1) & (h_channel > 1)) | (R > 1)] = 255
   
    # return combined hls image 
    return hls_comb


def combine_grad_hls(grad, hls):
    """ 
    #---------------------
    # This function combines gradient and hls images into one.
    # For binary gradient image, if pixel is bright, set that pixel value in reulting image to 255
    # For binary hls image, if pixel is bright, set that pixel value in resulting image to 255 
    # Edit: Assign different values to distinguish them
    # 
    """
    result = np.zeros_like(hls).astype(np.uint8)
    #result[((grad > 1) | (hls > 1))] = 255
    result[(grad > 1)] = 50
    result[(hls > 1)] = 255

    return result

def show_image(image, title="title", cmap_type="gray"):
    plt.imshow(image, cmap_type)
    plt.title(title)
    # plt.axis("off")
