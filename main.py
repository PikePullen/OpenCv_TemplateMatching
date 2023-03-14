import numpy as np
import matplotlib.pyplot as plt
import cv2

def display_img(img):
    plt.imshow(img)
    plt.show()

full = cv2.imread('../DATA/sammy.jpg')
full = cv2.cvtColor(full, cv2.COLOR_BGR2RGB)

# display_img(full)

face = cv2.imread('../DATA/sammy_face.jpg')
face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

# display_img(face)

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

# sample run, creates heatmap of most likely image match
# myMethod = eval('cv2.TM_CCOEFF')
# res = cv2.matchTemplate(full, face, myMethod)
# display_img(res)

for m in methods:
    # CREATE A COPY OF THE IMAGE
    full_copy = full.copy()

    """
    eval converts a string into a function, which is kind of cool
    myfunction = eval('sum')
    assertEquals(6, myfunction([1,2,3]))
    essentially the eval sum function turned it into a "sum" function
    passing an array of values performed the sum action on the array values
    """
    method = eval(m)

    # TEMPLATE MATCHING
    res = cv2.matchTemplate(full_copy, face, method)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # these methods have some differing exceptions on how they run
    if method in(cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
        top_left = min_loc
    else:
        top_left = max_loc

    height, width, channels = face.shape
    bottom_right = (top_left[0]+width, top_left[1]+height)

    cv2.rectangle(full_copy, top_left, bottom_right, (255,0,0), thickness=10)

    # plot images
    plt.subplot(121)
    plt.imshow(res)
    plt.title('HEATMAP OF TEMPLATE MATCHING')

    plt.subplot(122)
    plt.imshow(full_copy)
    plt.title('DETECTION OF TEMPLATE')

    # title of method used
    plt.suptitle(m)

    plt.show()
