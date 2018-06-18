import numpy as np
import PIL.Image as Image
import pdb

from PillowPrimitives import *

def Fuse(im1,im2):
    im1 = np.array(im1)
    im2 = np.array(im2)

    fusedim = im1 + im2

    fusedim[fusedim==35] = 20
    fusedim[fusedim==40] = 20

    fusedim = Image.fromarray(fusedim)
    return fusedim

def TranspFuse(im1,im2):
    im1 = np.array(im1)
    im2 = np.array(im2)

    fusedim = im1 + im2

    fusedim[fusedim==35] = 15
    fusedim[fusedim==40] = 20

    fusedim = Image.fromarray(fusedim)
    return fusedim

def Occlude(im1,im2):
    im1 = np.array(im1)
    im1[im1==15] = 16
    im1[im1==20] = 22
    im2 = np.array(im2)

    fusedim = im1 + im2

    fusedim[fusedim==30] = 15
    fusedim[fusedim==31] = 15
    fusedim[fusedim==40] = 20
    fusedim[fusedim==37] = 20
    fusedim[fusedim==36] = 15
    fusedim[fusedim==42] = 20

    fusedim[fusedim==16] = 15
    fusedim[fusedim==22] = 20

    fusedim = Image.fromarray(fusedim)
    return fusedim

def Rotate(im,theta):

    theta *= 360
    im = im.convert('RGBA')
    rot = im.rotate(theta) #resample = Image.BICUBIC)
    fff = Image.new('RGBA',rot.size,(255,)*4)
    im = Image.composite(rot,fff,rot)
    im = im.convert('L')
    return im

def ScaleX(im,scale):

    scale += 0.5 # range between 0.5 and 1.5
    im_w, im_h = im.size
    scaled_w = round(im_w * scale)
    scaledim = im.resize((scaled_w, im_h))#, resample = Image.BICUBIC)
    im = Image.new('L', (im_w, im_h), 255)
    offset = ((im_w - scaled_w) // 2, 0)
    im.paste(scaledim, offset)
    return im

def ScaleY(im,scale):

    scale += 0.5 # range between 0.5 and 1.5
    im_w, im_h = im.size
    scaled_h = round(im_h * scale)
    scaledim = im.resize((im_w, scaled_h))#, resample = Image.BICUBIC)
    im = Image.new('L', (im_w, im_h), 255)
    offset = (0, (im_h - scaled_h) // 2)
    im.paste(scaledim, offset)
    return im

def TranslateX(im,offset):

    # offset *= 100 # or whatever
    im_w, im_h = im.size
    shiftedim = Image.new('L', (im_w, im_h),255)
    shiftedim.paste(im, (offset,0))
    return shiftedim

Im1 = P3()
Im2 = P4()

Im2 = TranslateX(Im2,30)
FusedIm = Occlude(Im1,Im2)

FusedIm = Rotate(FusedIm,0.78)

#FusedIm = ScaleY(FusedIm, 0.1)
#FusedIm = ScaleX(FusedIm, 3.0)

#Im3 = Circle()

#FusedIm = Fuse(FusedIm,Im3)

FusedIm = np.array(FusedIm)

FusedIm[FusedIm==0] = 255
FusedIm[FusedIm==20] = 255

FusedIm = Image.fromarray(FusedIm)

FusedIm.show()
