import cv2
import math
import numpy as np
import sys
import matplotlib.pyplot as plt
def process_image(frame, centr, prev_left, struct=20): # 
    area_thresh = 8
    shoe_color = (94, 94, 202)

    # Compute the distance from the shoe color
    diff_from_shoe_color = (frame.astype(np.float32) - shoe_color)
    dist_from_shoe_color = np.linalg.norm(diff_from_shoe_color, axis=2)

    # Create a binary mask
    mask = (dist_from_shoe_color < 50)
    tmp = mask.astype(np.uint8) * 255

    # Morphological operations to clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (struct, struct))
    mask_dil = cv2.dilate(tmp, kernel)
    mask_final = cv2.erode(mask_dil, kernel)

    # Find contours and store as list
    contours, _ = cv2.findContours(mask_final, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contour_list = list(contours)

    # Sort contours by area and keep only the largest two in case our
    # morphological operators aren't perfect.
    if len(contour_list) > 2:
        # Compute areas and sort contours from largest to smallest
        sorted_contours = sorted(contour_list, key=lambda c: cv2.contourArea(c), reverse=True)

        # Keep only the two largest contours, discard the rest
        contour_list = sorted_contours[:2]
    # If a foot is obstructed, skip frame
    elif len(contour_list) < 2:
        return frame, tmp, mask_final, prev_left
    # Initialize the location of his left foot if this is the first frame
    if prev_left is None:
        prev_left = get_contour_info(contour_list[0])

    prev_left_mean = prev_left['mean']

    first_contour = get_contour_info(contour_list[0])
    second_contour = get_contour_info(contour_list[1])
    # Bug fix for small issue where one foot would pick up the second contour, 
    # but it was just a few pixels. Skip frame if this happens.
    if first_contour['area']>area_thresh and second_contour['area']>area_thresh:
        dist_from_left_foot = math.dist(first_contour['mean'], prev_left_mean)
        # Determine which of the two contours is his left foot
        if math.dist(second_contour['mean'], prev_left_mean) < dist_from_left_foot:
            contour_list[0], contour_list[1] = contour_list[1], contour_list[0]
    else:
        return frame, tmp, mask_final, prev_left

    # Draw contours onto the current frame
    for i, contour in enumerate(contour_list):
        info = get_contour_info(contour)                
        mu = info['mean']
        if frame is not None and mu is not None:
            cv2.drawContours(frame, [contour], -1, CONTOUR_COLORS[i], 2)
            centr.append(mu)

    return frame, tmp, mask_final, get_contour_info(contour_list[0])

def main():

    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print("please run program in format: program filename [width of structuring rectangle]")
        exit()

    video_path = sys.argv[1]
    video = cv2.VideoCapture(video_path)
   
    video.set(cv2.CAP_PROP_POS_FRAMES, 0) # Resets the video to frame 0
    # List to store all the centorids of the contours of every frame
    centr = []
    # Keep track of location of left foot so we can track the feet appropriately
    left = None
    # Loop through every frame of video
    while True:
        ok, frame = video.read()
        # Break out of loop when we have read past the last frame
        if not ok:
            break
        # Perform image processing on each frame to obtain outlines and position
        # runner's feet.
        if len(sys.argv) < 3:
            final_frame, mask, mask_final, left = process_image(frame, centr, left)
        else:
            final_frame, mask, mask_final, left = process_image(frame, centr, left, int(sys.argv[2]))

        #Show video play displaying each frame
        cv2.imshow('frame', final_frame)
        if cv2.waitKey(1) == ord('q'):
            break
        cv2.imshow('mask before morphological operators', mask)
        if cv2.waitKey(1) == ord('q'):
            break
        cv2.imshow('mask after morphological operators', mask_final)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyWindow('mask before morphological operators')
    cv2.destroyWindow('mask after morphological operators')
    cv2.destroyWindow('frame')
        
    x=[]
    y=[]
    # Plot left foot
    for i in range(0, len(centr), 2):
        if centr[i][0] is not None and centr[i][1] is not None:
            x.append(centr[i][0])
            y.append(centr[i][1])
    plt.plot(x,y, 'o', color="red", linestyle='-')

    x=[]
    y=[]
    # Plot right foot
    for i in range(1, len(centr), 2):
        if centr[i][0] is not None and centr[i][1] is not None:
            x.append(centr[i][0])
            y.append(centr[i][1])
    plt.plot(x,y, 'o', color="blue", linestyle='-')

    # Have y-axis start at lower values and increase moving down
    # Code from: https://stackoverflow.com/questions/2051744/how-to-invert-the-x-or-y-axis
    plt.gca().invert_yaxis()
    plt.show()

#Code used from Project 1 starter code. Attributed to Prof. Matt Zucker
def get_contour_info(c):

    # For more info, see
    #  - https://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html
    #  - https://en.wikipedia.org/wiki/Image_moment

    m = cv2.moments(c)

    s00 = m['m00']
    s10 = m['m10']
    s01 = m['m01']
    c20 = m['mu20']
    c11 = m['mu11']
    c02 = m['mu02']

    if s00 != 0:

        mx = s10 / s00
        my = s01 / s00

        A = np.array( [
                [ c20 / s00 , c11 / s00 ],
                [ c11 / s00 , c02 / s00 ] 
                ] )

        W, U, Vt = cv2.SVDecomp(A)

        ul = 2 * np.sqrt(W[0,0])
        vl = 2 * np.sqrt(W[1,0])

        ux = ul * U[0, 0]
        uy = ul * U[1, 0]

        vx = vl * U[0, 1]
        vy = vl * U[1, 1]

        mean = np.array([mx, my])
        uvec = np.array([ux, uy])
        vvec = np.array([vx, vy])

    else:
        
        mean = c[0].astype('float')
        uvec = np.array([1.0, 0.0])
        vvec = np.array([0.0, 1.0])

    return {'moments': m, 
            'area': s00, 
            'mean': mean,
            'b1': uvec,
            'b2': vvec}

CONTOUR_COLORS = [
    (255,   0,   0),
    (0,  0,   255),
]

main()
