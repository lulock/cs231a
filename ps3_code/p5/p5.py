import numpy as np
import cv2


def draw_tracks(frame_num, frame, mask, points_prev, points_curr, color):
    """Draw the tracks and create an image.
    """
    for i, (p_prev, p_curr) in enumerate(zip(points_prev, points_curr)):
        a, b = p_curr.ravel()
        c, d = p_prev.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(
            frame, (a, b), 3, color[i].tolist(), -1)

    img = cv2.add(frame, mask)

    cv2.imwrite('frame_%d.png'%frame_num,img)
    return img


def Q5_A():
    """Code for question 5a.

    Output:
      p0, p1, p2: (N,2) list of numpy arrays representing the pixel coordinates of the
      tracked features.  Include the visualization and your answer to the
      questions in the separate PDF.
    """
    # params for ShiTomasi corner detection
    feature_params = dict(
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(
        winSize=(75, 75),
        maxLevel=1,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.01),
        flags=(cv2.OPTFLOW_LK_GET_MIN_EIGENVALS))

    # Read the frames.
    frame1 = cv2.imread('p5_data/rgb1.png')
    frame2 = cv2.imread('p5_data/rgb2.png')
    frame3 = cv2.imread('p5_data/rgb3.png')
    frames = [frame1, frame2, frame3]

    # Convert to gray images.
    old_frame = frames[0]
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create some random colors for drawing
    color = np.random.randint(0, 255, (200, 3))

    # Create a mask image for drawing purposes
    mask = np.zeros_like(frame1)

    for i,frame in enumerate(frames[1:]):
        frame_gray = cv2.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # TODO: Fill in this code
        # BEGIN YOUR CODE HERE
        pass
        
        #Once you compute the new feature points for this frame, comment this out
        #to save images for your PDF:
        #draw_tracks(frame_num, frame, mask, points_prev, points_curr, color)
        # END YOUR CODE HERE

    return p0, p1, p2


def Q5_B(p0, p1, p2, intrinsic):
    """Code for question 5b.

    Note that depth maps contain NaN values.
    Features that have NaN depth value in any of the frames should be excluded
    in the result.

    Input:
      p0, p1, p2: (N,2) numpy arrays, the results from Q2_A.
      intrinsic: (3,3) numpy array representing the camera intrinsic.

    Output:
      p0, p1, p2: (N,3) numpy arrays, the 3D positions of the tracked features
      in each frame.
    """
    depth0 = np.loadtxt('p5_data/depth1.txt')
    depth1 = np.loadtxt('p5_data/depth2.txt')
    depth2 = np.loadtxt('p5_data/depth3.txt')

    # TODO: Fill in this code
    # BEGIN YOUR CODE HERE
    pass
    # END YOUR CODE HERE

    return p0, p1, p2


if __name__ == "__main__":
    p0, p1, p2 = Q5_A()
    intrinsic = np.array([[486, 0, 318.5],
                          [0, 491, 237],
                          [0, 0, 1]])
    p0, p1, p2 = Q5_B(p0, p1, p2, intrinsic)
