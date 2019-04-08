
from PIL import ImageFont, ImageDraw, Image 
import cv2
import numpy as np
from scipy.ndimage import rotate
from scipy.ndimage.filters import gaussian_filter
import skimage
import string
import random

#import pyvips

#https://stackoverflow.com/a/47381058/1018830
def trapez(y,y0,w):
    return np.clip(np.minimum(y+1+w/2-y0, -y+1+w/2+y0),0,1)

def weighted_line(r0, c0, r1, c1, w, rmin=0, rmax=np.inf, cmin=0, cmax=np.inf):
    # The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
    # If either of these cases are violated, do some switches.
    if abs(c1-c0) < abs(r1-r0):
        # Switch x and y, and switch again when returning.
        xx, yy, val = weighted_line(c0, r0, c1, r1, w, cmin,cmax,rmin,rmax)
        return (yy, xx, val)

    # At this point we know that the distance in columns (x) is greater
    # than that in rows (y). Possibly one more switch if c0 > c1.
    if c0 > c1:
        return weighted_line(r1, c1, r0, c0, w, rmin,rmax,cmin,cmax)

    # The following is now always < 1 in abs
    slope = (r1-r0) / (c1-c0)

    # Adjust weight by the slope
    w *= np.sqrt(1+np.abs(slope)) / 2

    # We write y as a function of x, because the slope is always <= 1
    # (in absolute value)
    x = np.arange(c0, c1+1, dtype=float)
    y = x * slope + (c1*r0-c0*r1) / (c1-c0)

    # Now instead of 2 values for y, we have 2*np.ceil(w/2).
    # All values are 1 except the upmost and bottommost.
    thickness = np.ceil(w/2)
    yy = (np.floor(y).reshape(-1,1) + np.arange(-thickness-1,thickness+2).reshape(1,-1))
    xx = np.repeat(x, yy.shape[1])
    vals = trapez(yy, y.reshape(-1,1), w).flatten()

    yy = yy.flatten()

    # Exclude useless parts and those outside of the interval
    # to avoid parts outside of the picture
    mask = np.logical_and.reduce((yy >= rmin, yy < rmax, xx>=cmin, xx<cmax, vals > 0))

    return (yy[mask].astype(int), xx[mask].astype(int), vals[mask])

#https://stackoverflow.com/a/47269413/1018830
def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)
def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)



class SyntheticText:

    def __init__(self,pad=20,line_prob=0.1,line_thickness=3,line_var=20,rot=10, gaus_noise=0.1, blur_size=1, hole_prob=0.2,hole_size=100,neighbor_gap_mean=20,neighbor_gap_var=7):
        self.line_prob = line_prob
        self.pad = pad
        self.line_thickness = line_thickness
        self.line_var = line_var
        self.rot=rot
        self.gaus=gaus_noise
        self.blur_size=blur_size
        self.hole_prob=hole_prob
        self.hole_size=hole_size
        self.neighbor_gap_mean=neighbor_gap_mean
        self.neighbor_gap_var=neighbor_gap_var

    def getText(self):
        l = np.random.randint(1,20)
        s = ''
        for i in range(l):
            s += random.choice(string.ascii_letters)
        return s

    def getFont(self):
        #return 'samanta'
        font = ImageFont.truetype("/home/ubuntu/.fonts/samantha.ttf", 100) 
        return font

    def getRenderedText(self,font=None,ink=None):
        random_text = self.getText()
        if font is None:
            font = self.getFont()

        #create big canvas as it's hard to predict how large font will render
        size=(250+190*len(random_text),920)
        image = Image.new(mode='L', size=size)

        draw = ImageDraw.Draw(image)
        if ink is None:
            ink=(np.random.random()/2)+0.5
        draw.text((400, 250), random_text, font=font,fill=1)
        np_image = np.array(image)

        horzP = np.max(np_image,axis=0)
        minX=first_nonzero(horzP,0)
        maxX=last_nonzero(horzP,0)
        vertP = np.max(np_image,axis=1)
        minY=first_nonzero(vertP,0)
        maxY=last_nonzero(vertP,0)
        return np_image,random_text,minX,maxX,minY,maxY,font,ink
        

    def getSample(self):
        np_image,random_text,minX,maxX,minY,maxY,font,ink = self.getRenderedText()
        if np.random.random()<1.1: #above
            if  np.random.random()<0.5:
                fontA=font
            else:
                fontA=None
            np_imageA,random_textA,minXA,maxXA,minYA,maxYA,_,_ = self.getRenderedText(fontA,ink)
            gap = np.random.normal(self.neighbor_gap_mean,self.neighbor_gap_var)
            moveA = int(minY-gap)-maxYA
            mainY1=max(0,minYA+moveA)
            mainY2=maxYA+moveA
            AY1=maxYA-(mainY2-mainY1)
            AY2=maxYA
            AxOff = np.random.normal(0,self.neighbor_gap_var*5)
            mainCenter = (maxX+minX)//2
            ACenter = (maxXA+minXA)//2
            AxOff = int((mainCenter-ACenter)+AxOff)
            mainX1 = max(0,minXA+AxOff)
            mainX2 = min(np_image.shape[1]-1,maxXA+AxOff)
            AX1 = minXA-(minXA+AxOff-mainX1)
            AX2 = maxXA-(maxXA+AxOff-mainX2)
            #print('[{}:{},{}:{}] [{}:{},{}:{}]'.format(mainY1,mainY2+1,mainX1,mainX2+1,AY1,AY2+1,AX1,AX2+1))
            np_image[mainY1:mainY2+1,mainX1:mainX2+1] = np.maximum(np_image[mainY1:mainY2+1,mainX1:mainX2+1],np_imageA[AY1:AY2+1,AX1:AX2+1])
        if np.random.random()<1.1: #below
            if  np.random.random()<0.5:
                fontA=font
            else:
                fontA=None
            np_imageA,random_textA,minXA,maxXA,minYA,maxYA,_,_ = self.getRenderedText(fontA,ink)
            gap = np.random.normal(self.neighbor_gap_mean,self.neighbor_gap_var)
            moveA = int(maxY+gap)-minYA
            mainY1=minYA+moveA
            mainY2=min(np_image.shape[0]-1,maxYA+moveA)
            AY1=minYA
            AY2=minYA+(mainY2-mainY1)
            AxOff = np.random.normal(0,self.neighbor_gap_var*5)
            mainCenter = (maxX+minX)//2
            ACenter = (maxXA+minXA)//2
            AxOff = int((mainCenter-ACenter)+AxOff)
            mainX1 = max(0,minXA+AxOff)
            mainX2 = min(np_image.shape[1]-1,maxXA+AxOff)
            AX1 = minXA-(minXA+AxOff-mainX1)
            AX2 = maxXA-(maxXA+AxOff-mainX2)
            #print('[{}:{},{}:{}] [{}:{},{}:{}]'.format(mainY1,mainY2+1,mainX1,mainX2+1,AY1,AY2+1,AX1,AX2+1))
            np_image[mainY1:mainY2+1,mainX1:mainX2+1] = np.maximum(np_image[mainY1:mainY2+1,mainX1:mainX2+1],np_imageA[AY1:AY2+1,AX1:AX2+1])


        #base_image = pyvips.Image.text(random_text, dpi=300, font=random_font)
        #org_h = base_image.height
        #org_w = base_image.width
        #np_image = np.ndarray(buffer=base_image.write_to_memory(),
        #        dtype=format_to_dtype[base_image.format],
        #        shape=[base_image.height, base_image.width, base_image.bands])


        #Distracting text
        #TODO
        np_image=np_image*0.8
        padding=np.random.randint(-self.pad//2,self.pad,(2,2))

        #lines
        while np.random.rand() < self.line_prob:
            side = np.random.choice([1,2,3,4])
            if side==1: #bot
                y1 = np.random.normal(maxY+20,self.line_var)
                y2 = y1+np.random.normal(0,self.line_var/4)
                x1 = np.random.normal(minX-padding[1,0],self.line_var*2)
                x2 = np.random.normal(maxX+padding[1,1],self.line_var*2)
            elif side==2: #top
                y1 = np.random.normal(minY-20,self.line_var)
                y2 = y1+np.random.normal(0,self.line_var/4)
                x1 = np.random.normal(minX-padding[1,0],self.line_var*2)
                x2 = np.random.normal(maxX+padding[1,1],self.line_var*2)
            elif side==3: #left
                x1 = np.random.normal(minX-20,self.line_var)
                x2 = x1+np.random.normal(0,self.line_var/4)
                y1 = np.random.normal(minY-padding[0,0],self.line_var*2)
                y2 = np.random.normal(maxY+padding[0,1],self.line_var*2)
            elif side==4: #right
                x1 = np.random.normal(maxX+20,self.line_var)
                x2 = x1+np.random.normal(0,self.line_var/4)
                y1 = np.random.normal(minY-padding[0,0],self.line_var*2)
                y2 = np.random.normal(maxY+padding[0,1],self.line_var*2)
            thickness = np.random.random()*(self.line_thickness-1) + 1
            yy,xx,val = weighted_line(y1,x1,y2,x2,thickness,0,np_image.shape[0],0,np_image.shape[1])
            color = np.random.random()
            np_image[yy,xx]=np.maximum(val*color,np_image[yy,xx])
            #print('line {}:  {},{}  {},{}'.format(side,x1,y1,x2,y2))

        #rot
        degrees=np.random.randint(-self.rot,self.rot)
        np_image = rotate(np_image,degrees,reshape=False)

        #crop
        #np_image = np.pad(np_image,padding,mode='constant')*0.8
        minY = max(0,minY-padding[0,0])
        minX = max(0,minX-padding[1,0])
        maxY = maxY+1+padding[0,1]
        maxX = maxX+1+padding[1,1]
        np_image = np_image[minY:maxY,minX:maxX]

        #holes
        while np.random.rand() < self.hole_prob:
            x=np.random.randint(0,np_image.shape[1])
            y=np.random.randint(0,np_image.shape[0])
            rad = np.random.randint(1,self.hole_size)
            rad2 = np.random.randint(rad/3,rad)
            size = rad*rad2
            rot = np.random.random()*2*np.pi
            strength = (1.6*np.random.random()-1.0)*(1-size/(self.hole_size*self.hole_size))
            yy,xx = skimage.draw.ellipse(y, x, rad, rad2, shape=np_image.shape, rotation=rot)
            complete = np.random.random()
            app = np.maximum(1-np.abs(np.random.normal(0,1-complete,yy.shape)),0)
            np_image[yy,xx] = np.maximum(np.minimum(np_image[yy,xx]+strength*app,1),0)

        #noise
        #specle noise
        #gaus_n = 0.2+(self.gaus-0.2)*np.random.random()
        gaus_n = np.random.normal(self.gaus,0.1)
        np_image += np.random.normal(0,gaus_n,np_image.shape)
        #blur
        blur_s = np.random.normal(self.blur_size,0.2)
        np_image = gaussian_filter(np_image,blur_s)

        #contrast/brighness

        return np_image, random_text
