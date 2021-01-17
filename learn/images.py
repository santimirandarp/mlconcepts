"""
1. Load Image File (image libraries)
2. Get the RGB representation
3. Feed to a "show" function

When using h5 files, step 1 is done by h5
It converts to IMG to RGB automatically.

We can use matplotlib.
(we could use img libraries though)
"""
from random import randint
from glob import glob

#setup
pathToFiles = "../images/raw/cats/*"
cats = glob(pathToFiles)
One = cats[randint(0, len(cats)-1)] #random cat

# Using convert to (cv2)
import cv2
#cv2 Docs https://tinyurl.com/y5ejrtmg
catImg = cv2.imread(One, 1)
cv2.imshow("random cat", catImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Playing a bit
color = cv2.imread(One, 1) #returns narray
grey = cv2.imread(One, 0)
red = cv2.imread(One, 1)
red[:,:,0:2] = 0 #set green and blue to zero
for cat in [color, grey, red]:
    cv2.imshow("random cat", cat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Show using matplot lib
import matplotlib.pyplot as plt
plt.imshow(color) #show
plt.show() #execute
