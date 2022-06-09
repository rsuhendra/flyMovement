# from PIL import Image, ImageDraw
# import imageio,sys

# filename = "../testFiles/v1.mp4"
# vid = imageio.get_reader(filename,  'ffmpeg')
# im  = Image.fromarray(vid.get_data(0))
# # filename = '/home/josh/Downloads/listParts.jpg'
# # im = Image.open(filename)
# draw = ImageDraw.Draw(im)
# draw.line((0, 0) + im.size, fill=128)
# draw.line((0, im.size[1], im.size[0], 0), fill=128)
# del draw

# im.show()
# # write to stdout
# #im.save(sys.stdout, "PNG")

#from __future__ import print_function
"""
Do a mouseclick somewhere, move the mouse to some destination, release
the button.  This class gives click- and release-events and also draws
a line or a box from the click-point to the actual mouseposition
(within the same axes) until the button is released.  Within the
method 'self.ignore()' it is checked whether the button from eventpress
and eventrelease are the same.

"""
from matplotlib.widgets import EllipseSelector
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import imageio
from tkinter import *
import sys
import tkinter.filedialog

# can remove this if not on mac
matplotlib.use("TkAgg")

def line_select_callback(eclick, erelease):
    # x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    #print(" The button you used were: %s %s" % (eclick.button, erelease.button))


def toggle_selector(event):
    print(' Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print(' Selector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print(' Selector activated.')
        toggle_selector.RS.set_active(True)

fig, current_ax = plt.subplots()                 # make a new plotting range
N = 100000                                       # If N is large one can see
x = np.linspace(0.0, 10.0, N)                    # improvement by use blitting!

filename = str(sys.argv[1])
outputName = str(sys.argv[2])
current_ax.set_title("Select the desired region of the arena")


vid = imageio.get_reader(filename,  'ffmpeg')
img = vid.get_data(0)

plt.imshow(img)
# plt.plot(x, +np.sin(.2*np.pi*x), lw=3.5, c='b', alpha=.7)  # plot something
# plt.plot(x, +np.cos(.2*np.pi*x), lw=3.5, c='r', alpha=.5)
# plt.plot(x, -np.sin(.2*np.pi*x), lw=3.5, c='g', alpha=.3)

print("Select the boundaries of the arena:")

# drawtype is 'box' or 'line' or 'none'
toggle_selector.RS = EllipseSelector(current_ax, line_select_callback,
                                       drawtype='box',useblit=False,
                                       button=[1, 3],  # don't use middle button
                                       minspanx=5, minspany=5,
                                       spancoords='pixels',
                                       interactive=True)
plt.connect('key_press_event', toggle_selector)
plt.show()
rcenter = toggle_selector.RS.center
redges = toggle_selector.RS.extents

width = redges[1]-redges[0]
height = redges[3]-redges[2]

print ("Center:", rcenter)
print ("Width:", width)
print ("Height:", height)

class LineBuilder:
    def __init__(self, line):
        self.line = line
        self.xs = [] # list(line.get_xdata())
        self.ys = [] #list(line.get_ydata())
        self.xsOld = []
        self.ysOld = []
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        #print('click', event)
        if event.inaxes!=self.line.axes: return

        if len(self.xs) %2 ==1:
          self.xs.append(event.xdata)
          self.ys.append(event.ydata)
          self.line.set_data(self.xs, self.ys)
          self.line.figure.canvas.draw()
          self.xsOld = self.xs
          self.ysOld = self.ys
          self.xs = []
          self.ys =[]
        else:
          self.xs.append(event.xdata)
          self.ys.append(event.ydata)
          self.line.set_data(self.xs, self.ys)
          self.line.figure.canvas.draw()
          self.xsOld = self.xs
          self.ysOld = self.ys


fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(img)
ax.set_title('Draw vertical division')
line, = ax.plot([0], [0])  # empty line
line2 = LineBuilder(line)

plt.show()
t1 = line2.xsOld, line2.ysOld

fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(img)
ax.set_title('Draw horizontal division')
line, = ax.plot([0], [0])  # empty line
line2 = LineBuilder(line)

plt.show()
t2 = line2.xsOld, line2.ysOld

data = (rcenter,width,height,t1,t2)

import pickle
outputName = outputName + ".arena"
pickle.dump(data,open('arenas/'+outputName,"wb"))
