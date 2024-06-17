import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

LOCATION_VEHICLE_PIXEL = (700, 400)
PIXELS_PER_METER = 10

def save_video_from_images(filename, images, fps):
    _image = images[0]
    image_shape = {'height': _image.shape[0],
                   'width': _image.shape[1],
                   'layers': _image.shape[2]}

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(filename, fourcc, fps,
                            (image_shape['width'], image_shape['height']))
    for image in images:
        video.write(image)
    video.release()
    
def save_episode_plot(x_array, y_array, xlabel, ylabel, vertical_line, filename):

    fig = plt.figure()
    
    running_avg = y_array

    plt.plot(x_array, running_avg)
    plt.xlabel(f"{xlabel}")
    plt.ylabel(f"{ylabel}")
    
    if vertical_line != None:
        plt.axvline(x=int(vertical_line), color='g')
        plt.legend([f'{ylabel}','model saved'])
    else:
        plt.legend([f'{ylabel}'])
    
    plt.savefig(filename)
    plt.close(fig)
    
    