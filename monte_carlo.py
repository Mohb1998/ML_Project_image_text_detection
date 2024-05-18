import time
import random
import os
import numpy as np
#import pandas as pd
from predict import extract_and_predict 
import joblib
from tensorflow.keras.models import load_model
random.seed(42)


def compare_strings(str1, str2):
       min_length = min(len(str1), len(str2))
       comparisons = []
       
       for i in range(min_length):
              comparisons.append(str1[i] == str2[i])
       return comparisons


image_dir = "./input/test_v2/test"

image_files = [os.path.join(image_dir, os.path.normpath(file)) for file in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, file))]
#image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, file))]

encoder = joblib.load("ML_Project_image_text_detection/label_encoder.pkl")
cnn_model = load_model("ML_Project_image_text_detection/trained_model.h5")

texts = np.array([], dtype=str) 
times = np.array([], dtype=float) 

when_break = 20
i = 0 
for file in image_files:
       result = extract_and_predict(file, cnn_model=cnn_model, encoder=encoder)
       texts = np.append(texts, result[0])
       times = np.append(times, result[2])
       print(file)
       i = i + 1
       if (i > when_break):
              break

print(texts)
print(times)

texts_correct = np.loadtxt('./input/written_name_test_v2.csv', delimiter=",", dtype=str)
for k in range(1,when_break+1):
       print(texts_correct[[k]][0][1]) # texts_correct[[i]][0][1] how to get to the correct result 
       print(sum(compare_strings(texts_correct[[k]][0][1], texts[k])))















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


