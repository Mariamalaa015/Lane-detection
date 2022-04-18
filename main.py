import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from moviepy.editor import VideoFileClip
import glob
import matplotlib.image as mpimg
from test import get_combined_gradients, get_combined_hls, combine_grad_hls

#image = cv2.imread("challenge_25.jpg")

#height, width, _ = image.shape

image_files = glob.glob('camera_cal/calibration*.jpg')

"""
Implementation Notes:
--------------------
We need to map Image Points to Object Points.
Image Points: The coordinates of the corners in these 2D images 
- To get image points, I am using cv2.findChessboardCorners.
Object Points: The 3D coordinates of real undistorted chess board corners
- The object points are known; they are the known coordinates of the chessboard corners for a 9x6 board. 
- This points will be 3D coordinates.
- For an 9x6 board:
     Top left corner = (0,0,0)
     Bottom right corner = (8,5,0)
     The z, in (x,y,z) is 0 for all corners, since the chessboard is a flat 2D surface.
"""

# Array to store object points and image points from all the images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

def calib():
    """
    #--------------------
    # To get an undistorted image, we need camera matrix & distortion coefficient
    # Calculate them with 9*6 20 chessboard images
    #
    """

    # Prepare object points
    # This object points will be the same for all calibration images
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)  # x,y coordinates

    for curr_file in image_files:

        img = mpimg.imread(curr_file)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If corners are found, add object points, image points
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
        else:
            continue

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return mtx, dist

mtx, dist = calib()

def undistort(img, mtx, dist):
    """ 
    #--------------------
    # undistort image 
    #
    """
    return cv2.undistort(img, mtx, dist, None, mtx)


region_of_interest_vertices = [
    (0, height),
    (width / 2,400),
    (width, height),
]

def show_image(image, title="title", cmap_type="gray"):
    plt.imshow(image, cmap_type)
    plt.title(title)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    
    cv2.fillPoly(mask, np.array([vertices], np.int32), match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def get_src_dst(image):
    src = [595, 452], \
          [685, 452], \
          [1110, image.shape[0]], \
          [220, image.shape[0]]
    line_dst_offset = 100
    dst = [src[3][0] + line_dst_offset, 0], \
            [src[2][0] - line_dst_offset, 0], \
            [src[2][0] - line_dst_offset, src[2][1]], \
            [src[3][0] + line_dst_offset, src[3][1]]
    
    return src, dst


#def bilateral_filter(image):
    #return cv2.bilateralFilter(image, 15, 75, 75)

#def histogram_filter(image):
 #   gray = cv2.cvtColor(bilateral_filter(image), cv2.COLOR_BGR2GRAY)
  #  gray[gray <= 110] = 150
   # return gray

def sobel(image):
    sobel_kernel=7
    mag_thresh=(3, 255)
    
    # gray = histogram_filter(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[:,:,1]
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = 0
    
    sobel_abs = np.abs(sobelx**2 + sobely**2)
    sobel_abs = np.uint8(255 * sobel_abs / np.max(sobel_abs))
        
    sobel_binary = np.zeros(shape=gray.shape, dtype=bool)
    sobel_binary[(sobel_abs > mag_thresh[0]) & (sobel_abs <= mag_thresh[1])] = 1

    return sobel_binary

def s_thresholding(image):
    s_thresh=(170, 255)
    
    hls = cv2.cvtColor(bilateral_filter(image), cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros(shape=s_channel.shape, dtype=bool)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    return s_binary

def combine_sobel_and_thresholding(image):
    img1 = sobel(image)
    img2 = s_thresholding(image)
    
    combined_binary = np.zeros_like(img1).astype(np.uint8)
    combined_binary[(img1 == 1) | (img2 == 1)] = 1

    combined_binary = np.uint8(255 * combined_binary / np.max(combined_binary))
    return combined_binary

def masked_image(combined_binary):
    offset = 100
    mask_poly =np.array([[(0 + offset, image.shape[0]),
                            (image.shape[1] / 2.3, image.shape[0] / 1.65),
                            (image.shape[1] / 1.7, image.shape[0] / 1.65),
                            (image.shape[1], image.shape[0])]],
                          dtype=int)
    mask_img = np.zeros_like(combined_binary)
    ignore_mask_color = 255
    cv2.fillPoly(mask_img, mask_poly, ignore_mask_color)
    masked_edges = cv2.bitwise_and(combined_binary, mask_img)
    return masked_edges

def warp(image, src, dst):

    src = np.float32([src])
    dst = np.float32([dst])
    
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(src, dst),
                               dsize=image.shape[0:2][::-1], flags=cv2.INTER_LINEAR)

def get_warped_image_histogram(image):
    return np.sum(image[image.shape[0] // 2:, :], axis=0)

def sliding_window(image, n_windows=9):
    # Get Binary image histogram
    # histogram = get_warped_image_histogram(image)
    
    # Allow image to be RGB to display sliding windows.
    out = np.dstack((image, image, image)) * 255
    
    # midpoint = histogram.shape[0] // 2

    # Get Center of left and right peaks.
    # leftx_base = np.argmax(histogram[250:midpoint]) + 350
    # rightx_base = np.argmax(histogram[midpoint:-250]) + midpoint
    # print(leftx_base, rightx_base)
    leftx_base = 350
    rightx_base = 950

    # Calculate window height.
    window_height = image.shape[0] // n_windows
    
    # Get indicies that map to non zero values.
    non_zero_y, non_zero_x = map(lambda x: np.array(x), image.nonzero())
    
    margin = 50 # Width of sliding window.
    min_pixels = 50 # Min amount of pixels that match inside the window to detect a lane.
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    left_lane_indicies = []
    right_lane_indicies = []
    
    for window in range(n_windows):
        # Calculate window vertices positions.
        win_y_low = image.shape[0] - (window + 1) * window_height # Y coordinate of top corners. 
        win_y_high = image.shape[0] - window * window_height # Y coordinate of bottom corners.
        win_xleft_low = leftx_current - margin # X coordinate of left side of the window at the left side of the lane.
        win_xleft_high = leftx_current + margin # X coordinate of right side of the window at the left side of the lane.
        win_xright_low = rightx_current - margin # X coordinate of left side of the window at the right side of the lane.
        win_xright_high = rightx_current + margin # X coordinate of right side of the window at the right side of the lane.
        
        # Draw green rectangle at current windows.
        cv2.rectangle(out, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        
        good_left_indicies = ((non_zero_y >= win_y_low) & (non_zero_y < win_y_high) & (non_zero_x >= win_xleft_low) & (
            non_zero_x < win_xleft_high)).nonzero()[0] # Check if index is inside the left window.
        good_right_indicies = ((non_zero_y >= win_y_low) & (non_zero_y < win_y_high) & (non_zero_x >= win_xright_low) & (
            non_zero_x < win_xright_high)).nonzero()[0] # Check if index is inside the right window.
        
        left_lane_indicies.append(good_left_indicies)
        right_lane_indicies.append(good_right_indicies)
        
        # Shift left to the mean.
        if len(good_left_indicies) > min_pixels:
            leftx_current = int(np.mean(non_zero_x[good_left_indicies]))
        if len(good_right_indicies) > min_pixels:
            rightx_current = int(np.mean(non_zero_x[good_right_indicies]))
    
    left_lane_indicies = np.concatenate(left_lane_indicies)
    right_lane_indicies = np.concatenate(right_lane_indicies)

    # Pixels that matched in the left side of the lane.
    left_x = non_zero_x[left_lane_indicies]
    left_y = non_zero_y[left_lane_indicies]
    
    # Pixels that matched in the right side of the lane.
    right_x = non_zero_x[right_lane_indicies]
    right_y = non_zero_y[right_lane_indicies]

    # Fit the points using a second degree polynomial.
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)
        
    return out, left_fit, right_fit
def draw_lines_and_fill(image, warped_image, left_fit, right_fit, src, dest):
    
    # Make a zero like copy of warped image.
    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    
    # Make rgb image of zeros.
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Create a y axis.
    ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
    ploty = ploty[ploty.shape[0] // 2:] # zwdt dh
        
    # Left line polynomial.
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    # Right Line polynomial.
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    #cv2.fillPoly(color_warp_center, np.int_([pts]), (0, 255, 0))
    cv2.fillPoly(color_warp, np.int_([pts]), (255, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = warp(color_warp, dest, src)
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.2, 0)

    color_warp_lines = np.dstack((warp_zero, warp_zero, warp_zero))
    cv2.polylines(color_warp_lines, np.int_([pts_left]), isClosed=False, color=(0, 0, 255), thickness=25)
    cv2.polylines(color_warp_lines, np.int_([pts_right]), isClosed=False, color=(0, 0, 255), thickness=25)
    newwarp_lines = warp(color_warp_lines, dest, src)

    result = cv2.addWeighted(result, 1, newwarp_lines, 1, 0)

    return result

def img_write(path, image):
    cv2.imwrite(path, image)

def pipeline(image):
    th_sobelx, th_sobely, th_mag, th_dir = (35, 100), (30, 255), (30, 255), (0.7, 1.3)
    th_h, th_l, th_s = (10, 100), (0, 60), (85, 255)

    # undistorted_image = undistort(image, mtx, dist)
    # undistorted_image = cv2.resize(undistorted_image, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)

    # img_write('1.png', undistorted_image)

    # combined_binary = combine_sobel_and_thresholding(image)

    # remove_dark_colors = image.copy()
    # remove_dark_colors[np.mean(image) <= 110] = np.array([150,150,150])
    
    # cv2.imshow('frame', remove_dark_colors)

    combined_gradient = get_combined_gradients(image, th_sobelx, th_sobely, th_mag, th_dir)

    combined_hls = get_combined_hls(image, th_h, th_l, th_s)

    combined_binary = combine_grad_hls(combined_gradient, combined_hls)

    img_write('2.png', combined_binary)

    src, dst = get_src_dst(image)

    warped_image = warp(combined_binary, src, dst)

    img_write('3.png', warped_image)

    out, left_fit, right_fit = sliding_window(warped_image)

    img_write('4.png', out)

    src, dst = get_src_dst(image)

    img_write('5.png', draw_lines_and_fill(image, warped_image, left_fit, right_fit, src, dst))

    return draw_lines_and_fill(image, warped_image, left_fit, right_fit, src, dst)

# white_output = f"./{time.time()}.mp4"
# frame = VideoFileClip('actual_challenge_video.mp4')
# white_clip = frame.fl_image(pipeline)
# white_clip.write_videofile(white_output, audio=False)

cap = cv2.VideoCapture('actual_challenge_video.mp4')

while cap.isOpened():
    ret, image = cap.read()

    if image is None:
        break

    cv2.imshow('frame', pipeline(image))

    # img_array.append(img)

    if cv2.waitKey(60) & 0xFF == ord('q'):
        break
