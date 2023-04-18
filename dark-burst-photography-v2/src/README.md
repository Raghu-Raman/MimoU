# dark-burst-photography-v2

The goal of this project is to build a low-light image enhancement network trained on yuv format that can be deployed on mobile phones. 

The network adopt muti-frame strategy which means different frames that have different exposure time are input to the model together.  The current setting is adopt three input frames where the exposure values are ev -20,  0 and 4. 


# Training

Install the packages in the requirement.txt then you should be ok to run the code.
The training dataset and test set will be shared with you by google drive link. You should move the dataset to the dataset folder then run the training code. The training dataset is large and is seperated into different google drive folder. You should download them and combine them into a single folder.

1. https://drive.google.com/file/d/1X2tXHo3x-W_-xnyZ98UpA4Lq4jO3Lms_/view?ts=632a2517
	This link contains 597 scenes for training.
2. https://drive.google.com/file/d/10NWjob5aA2jBM81K6Ke9Kb1prW_9ZVIN/view?ts=632a24b3
	 This link contains 459 scenes for training.
3. Another 800 images are on the AWS, you nay need to ask Joe or Vivek to get permission.

After downloading the data, you can run train_light_no_map_fn_no_norm_yuv_loader_no_std_cat.py to train the model.

# Test
Here are several test set.
https://drive.google.com/drive/u/1/folders/1OrAcIZ2FBUCWjZe3vorKYXdYsDkaL7en
    This link contains some outdoor scenes.

https://drive.google.com/file/d/1RtQXPGAWIvhMta-ICvFIHN9VbtByEmBY/view
	This link contains 11 test frames.
You can run test_v2_no_map_fn_yuv_loader_no_std.py to test the results.

# Tflite_converter
After training, you have a saved model. You need to run the tflite_converter.py file to convert the saved model to a tflite file, which can be test and run on a mobile phone. I will give you the devices when we meet in person.

You need to push the benchmarktool to the devices and benchmark the tflite model.
You can find the adb command instructions in instruction.txt.


# Memory goal

The goal of the current workstream is to reduce the overall memory cost of the tflite benchmarking to <700MB.  After the goal is achieved, this workstream will be paused and shift to working on the Penang Pilot w/MBG.
