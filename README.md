# Bolt-detection-and-localization
bolt detection and localization only from two images using image processing and deep learning

##Requirement:
1. pytorch
2. opencv3
3. numpy
4. PIL
5. skimages
6. matplotlib

##How to run:
Open the terminal in the switchon directory and run the defect_detector.py file with python followed by --imagepath with image path 
Example:
$python defect_detector.py --imagepath D:\switchon\main_data\good_image.png

class(withbolt or withoutbolt) of the image will be printed and if the image belongs to withbolt class then coordinates will also get printed.
Following all this image will pop up with the class writen on the heading of it. If the image belongs to the withbolt class then rectangular bounding box will also be shown in the image, press any key disapper the shown image and then the class and the coordinates will get printed.

##Additional information:
-> The two image you provided, I augmented it and make to 4860 images.
-> The classifier for cropped images(also augmented) I build is based on resnet34. Has accuracy about 92%
-> Tested on the 4860 augmented image the full pipeline give about 91% of accuracy.
-> The whole process took 1034 secs which means the full pipeline takes 0.21 secs to test single image after loading the model.
