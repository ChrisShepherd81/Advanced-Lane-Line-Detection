'''
Created on 05.02.2017

@author: christian
'''
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from Pipeline import pipeline

def process_image(image):
    return pipeline(image)

if __name__ == '__main__':
    videoName = "../project_video.mp4"
    #videoName = "challenge_video.mp4"
    #videoName = "harder_challenge_video.mp4"
    outputVideo = '../output_video/' + videoName

    clip = VideoFileClip("../" + videoName)
    clipFilter = clip.fl_image(process_image) 
    clipFilter.write_videofile(outputVideo, audio=False)