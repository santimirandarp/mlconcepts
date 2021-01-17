from random import randint
from glob import glob
#setup
pathToFiles = "../images/raw/cats/*"
cats = glob(pathToFiles)
One = cats[randint(0, len(cats)-1)]

# Using PIL (python image library)
from PIL import Image

catImg = Image.open(One) #load
catImg.show() #show

# Show using matplotlib
plt.imshow(catImg) #show
plt.show()

