Stereo vision algorithm using OpenCV2 stereoBM. 

stereoTesting.py was tested and tuned for the WSVD Stereo Video dataset, and stereoCamera.py was tested and tuned for the ELP GS800P-V83 USB Stereo camera.

  NOTES ON THE STEREO CAMERA:
    -Due to either the focal length or the baseline length between the two cameras, the camera has trouble seeing depth on objects closer then about 3 feet. Additionally, objects inside of this range appear
     'doubled' or with ghost sillhouttes. This happens both when the left and right images are passed through stereoBM, and when the two images are simply bitwise ANDed together through cv2.bitwise_and().
    -The algorithm has a sweet spot from about 3 to 5 feet where the output gradient is a bright white to gray to black. Further than that, and most things appear as black with noisy splotches of white. 
    -For some reason, only with the stereoCamera.py, the startup time of the script is about 15 seconds from running the script to seeing video.
