import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import math
import numpy as np

# primitives

def P(a):

    image = Image.new("L",(200,200),"black")

    draw = ImageDraw.Draw(image)

    UNIT_CIRCLE = [(math.sin(math.radians(a)), math.cos(math.radians(a)))
        for a in range(0, 360, int(round(360/a)))]

    for radius in range(22,18,-1):
        #radius = 20

        points = []
        for x,y in UNIT_CIRCLE:
            points += [x * radius + 100, y * radius + 100]
            #points += [x * radius + 20, y * radius + 20]

        draw.polygon(points, fill=20,outline=15)

    return image

def C():

    image = Image.new("L",(200,200),"black")
    draw = ImageDraw.Draw(image)
    for radius in range(-1,2,1):
        bbox = [80+radius,80+radius,120-radius,120-radius]
        #bbox = [0+radius,0+radius,40-radius,40-radius]
        draw.ellipse(bbox,fill=20,outline=15)
    return image

## transformations

def OC(im1,im2):
    im1 = np.array(im1)
    im1[im1<15] = 0
    im1[im1>15] = 20
    im1[im1==15] = 16
    im1[im1==20] = 22
    im2 = np.array(im2)
    im2[im2<15] = 0
    im2[im2>15] = 20

    fusedim = im1 + im2

    fusedim[fusedim==31] = 15
    fusedim[fusedim==37] = 20
    fusedim[fusedim==36] = 15
    fusedim[fusedim==42] = 20

    fusedim[fusedim==16] = 15
    fusedim[fusedim==22] = 20

    fusedim = Image.fromarray(fusedim)
    return fusedim

def R0(im,theta):

    theta *= 360
    im = im.convert('RGBA')
    rot = im.rotate(theta) #resample = Image.BICUBIC)
    fff = Image.new('RGBA',rot.size,(255,)*4)
    im = Image.composite(rot,fff,rot)
    im = im.convert('L')
    im = np.array(im)
    im[im<15] = 0
    im[im>15] = 20 
    im = Image.fromarray(im) 
    return im

def Sx(im,scale):

    scale = scale*2+0.5 # range between 0.5 and 2.5
    im_w, im_h = im.size
    scaled_w = round(im_w * scale)
    scaledim = im.resize((int(scaled_w), im_h)) #resample = Image.BICUBIC)
    im = Image.new('L', (im_w, im_h), 0)
    offset = (int((im_w - scaled_w) // 2), 0)
    im.paste(scaledim, offset)
    im = np.array(im)
    im[im<15] = 0
    im[im>15] = 20 
    im = Image.fromarray(im) 
    return im

def Sy(im,scale):

    scale = scale*2+0.5 # range between 0.5 and 2.5
    im_w, im_h = im.size
    scaled_h = round(im_h * scale)
    scaledim = im.resize((im_w, int(scaled_h))) #resample = Image.BICUBIC)
    im = Image.new('L', (im_w, im_h), 0)
    offset = (0, int((im_h - scaled_h) // 2))
    im.paste(scaledim, offset)
    im = np.array(im)
    im[im<15] = 0
    im[im>15] = 20 
    im = Image.fromarray(im) 
    return im

def Tx(im,offset):

    offset = offset*100 - 50 # or whatever
    im_w, im_h = im.size
    shiftedim = Image.new('L', (im_w, im_h),0)
    shiftedim.paste(im, (int(offset),0))
    shiftedim = np.array(shiftedim)
    shiftedim[shiftedim<15] = 0
    shiftedim[shiftedim>15] = 20 
    shiftedim = Image.fromarray(shiftedim)
    return shiftedim

def Ty(im,offset):

    offset = offset*100 - 50 # or whatever
    im_w, im_h = im.size
    shiftedim = Image.new('L', (im_w, im_h),0)
    shiftedim.paste(im, (0,int(offset)))
    shiftedim = np.array(shiftedim)
    shiftedim[shiftedim<15] = 0
    shiftedim[shiftedim>15] = 20 
    shiftedim = Image.fromarray(shiftedim)
    return shiftedim

def SF(im1,im2):
    im1 = np.array(im1)
    im2 = np.array(im2)

    im1[im1<15] = 0
    im1[im1>15] = 20
    im2[im2<15] = 0
    im2[im2>15] = 20

    fusedim = im1 + im2

    fusedim[fusedim==30] = 15
    fusedim[fusedim==35] = 20
    fusedim[fusedim==40] = 20

    fusedim = Image.fromarray(fusedim)
    return fusedim

def TF(im1,im2):
    im1 = np.array(im1)
    im2 = np.array(im2)

    im1[im1<15] = 0
    im1[im1>15] = 20
    im2[im2<15] = 0
    im2[im2>15] = 20

    fusedim = im1 + im2

    fusedim[fusedim==35] = 15
    fusedim[fusedim==30] = 15
    fusedim[fusedim==40] = 20

    fusedim = Image.fromarray(fusedim)
    return fusedim

#FusedIm = R(TF(OCCL(Tx(P(2),0.56),P(6)),Ty(P(3),.75)),.15)
#FusedIm = P(3)

#FusedIm = np.array(FusedIm)

#FusedIm[FusedIm==0] = 255
#FusedIm[FusedIm==20] = 255

#print(np.unique(FusedIm))

#FusedIm = Image.fromarray(FusedIm)

#FusedIm.show()
