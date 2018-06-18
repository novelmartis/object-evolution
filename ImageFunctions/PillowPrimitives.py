
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import math

def P2():

    image = Image.new("L",(200,200),"black")

    draw = ImageDraw.Draw(image)

    points = [90,100,110,100]
    draw.line(points, fill=15,width=1)

    return image

def P3():

    image = Image.new("L",(200,200),"black")

    draw = ImageDraw.Draw(image)

    points = ((100,90),(90,110),(110,110))
    draw.polygon(points, fill=20,outline=15)

    return image

def P4():

    image = Image.new("L",(200,200),"black")

    draw = ImageDraw.Draw(image)

    bbox = [90,90,110,110]
    #bbox = [105,105,125,125]
    draw.rectangle(bbox, fill=20,outline=15)

    return image

def P5():

    image = Image.new("L",(200,200),"black")

    draw = ImageDraw.Draw(image)

    UNIT_CIRCLE = [(math.sin(math.radians(a)), math.cos(math.radians(a)))
        for a in range(0, 360, round(360/5))]

    radius = 20

    points = []
    for x,y in UNIT_CIRCLE:
        points += [x * radius + 100, y * radius + 100]

    draw.polygon(points, fill=20,outline=15)

    return image

def Circle():

    image = Image.new("L",(200,200),"black")

    draw = ImageDraw.Draw(image)

    bbox = [90,90,110,110]
    draw.ellipse(bbox,fill=20,outline=15)

    return image

#Im2 = P3()
#Im2.show()
