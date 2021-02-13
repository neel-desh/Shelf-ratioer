from PIL import Image
import os
try:
    image = Image.open('demo.jpeg')
    image1 =image
    image2 = image
    width, height = image.size

    row1 = (0,0,width,height//3)

    row2 = (0,height//3,width,(height//3)*2)

    row3 = (0,(height//3)*2,width,height)

    filename  = 'static\cropped\saverow1.jpg'
    filename2 = 'static\cropped\saverow2.jpg'
    filename3 = 'static\cropped\saverow3.jpg'

    cropped_image = image.crop(row1)
    cropped_image.save(filename)

    cropped_image1 = image1.crop(row2)
    cropped_image1.save(filename2)

    cropped_image2 = image2.crop(row3)
    cropped_image2.save(filename3)

    ## Remove comments to auto delete the file
    #os.remove(filename)
    #os.remove(filename2)
    #os.remove(filename3)

except Exception:
    print("Some error occured")
