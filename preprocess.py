#To cmd D:/Anaconda/python.exe d:/Yazılım/image-clasiffier/preprocess.py
import os 
import logging 
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from smartcrop import detect
import numpy as np

logging.basicConfig(
    level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s"
)

def check_is_dir(path):
    if not os.path.isdir(path):
        raise ValueError(f"Provided {path} is not a directory") 
    return True

def filter_images(list_of_files):
    valid_extensions={'.jpg','.jpeg','.png','.webp'}
    return [file for file in list_of_files if any(file.endswith(ext) for  ext in valid_extensions)]

def load_image(image_path):
    image=Image.open(image_path).convert('RGB')
    return image

def read_images_from_dir(dir_path):
    check_is_dir(dir_path)
    files=os.listdir(dir_path)
    image_files=filter_images(files)
    image_paths=[os.path.join(dir_path,file) for file in image_files]
    images=[load_image(image_path) for image_path in tqdm(image_paths)]
    logging.info(f"Load {len(images)} images from {dir_path}")
    return images

loaded_images=read_images_from_dir("raw_images/bird")



def max_resolution_rescale(image,max_width,max_height):
    width, height = image.size
    if width> max_width or height > max_height:
        ratio = min(max_width/width , max_height/height )
        new_width = int(width*ratio)
        new_height = int(height*ratio)
        image=image.resize((new_width,new_height),Image.LANCZOS)
    return image

max_size =1024

resized_images=list(map(lambda x : max_resolution_rescale(x,max_size,max_size),tqdm(loaded_images)))

def min_resolution_filter(image,min_width,min_height):
    width,height= image.size
    return width>=min_width and height>=min_height

min_size=224
filtered_images = list(filter(lambda x: min_resolution_filter(x,min_size,min_size),tqdm(resized_images)))
print(len(filtered_images))

def plot_image(image):
    plt.imshow(image)
    plt.axis("off")
    plt.show()
plot_image(filtered_images[14])


#Cropping images with center_crop 
def center_crop(image,new_width, new_height):
    width,height=image.size
    left=(width-new_width)/2
    top=(height-new_height)/2
    right=(width+new_width)/2
    bottom=(height+new_height)/2
    croppped_image=image.crop((left,top,right,bottom))
    logging.info(f"Center cropped image to {new_width}x{new_height}")
    return croppped_image
min_size=224

plot_image(center_crop(filtered_images[14],min_size,min_size))

#Cropping images with smartcrop

cropped_images=list(map(lambda x: detect(x,square=True),tqdm(filtered_images)))