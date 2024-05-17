import PyAutoGUI
import time
import os

random.seed(42)

#path = "./input/test_v2/test"
 
#obj = os.scandir(path=path)



# Directory containing images
image_dir = "./input/test_v2/test"

image_files = [os.path.join(image_dir, os.path.normpath(file)) for file in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, file))]
#image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, file))]


#image_files = [os.path.abspath(os.path.abspath(os.path.join(image_dir, file))) for file in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, file))]
#image_dir, 

#print(image_files)

###### PSEUDOCODE ###### 

# for i in images 
##  load image
###   put it in the model - label binarizer 
##  save result, time, label = this is in the predicted_text.txt
## return lists of: results, correct labels, times

# accuracy = compare result x label 
# efficiency  = accuracy/time 

# mean(time, accuracy, efficiency) 

# repeat 1000 times (record the changes? are there any?)\

############ ############ 




n_times = 10  # Change this to the desired number of executions

pyautogui.click(x=100, y=200)
image_path = image_files[1]
pyautogui.write(image_path)
pyautogui.press('enter')
time.sleep(1) 


pyautogui.click(x=150, y=250)  # Change (x, y) to the coordinates of your 'again' button
time.sleep(1) 

#image_path = image_files[i % len(image_files)]  # Cycle through the images

#for i in range(n_times):
        # Press the 'next' button
 #       pyautogui.click(x=100, y=200)  # Change (x, y) to the coordinates of your 'next' button
  #      time.sleep(1)  # Adjust sleep time as necessary
        
        # Select an image file
 #       image_path = image_files[i]
        # Assuming the file selection dialog is open, type the image path and press Enter
 #       pyautogui.write(image_path)
   #     pyautogui.press('enter')
  #      time.sleep(1)  # Adjust sleep time as necessary
        
        # Get the result (assumed to happen automatically after image selection)
        # Press the 'again' button to restart the process
   #     pyautogui.click(x=150, y=250)  # Change (x, y) to the coordinates of your 'again' button
   #     time.sleep(1)  # Adjust sleep time as necessary





