import time
import random
import os
import numpy as np
#import pandas as pd
from predict import extract_and_predict 
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
random.seed(42)

def compare_strings(str1, str2):
       min_length = min(len(str1), len(str2))
       score = []
       correct = []
       mistaken = []
       
       for i in range(min_length):
              scores.append(str1[i] == str2[i])
              if str1[i] != str2[i]:
                     correct.append(str1[i])
                     mistaken.append(str2[i])
       return score, correct, mistaken

image_dir = "./input/test_v2/test"

image_files = [os.path.join(image_dir, os.path.normpath(file)) for file in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, file))]
#image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, file))]

encoder = joblib.load("ML_Project_image_text_detection/label_encoder.pkl")
cnn_model = load_model("ML_Project_image_text_detection/trained_model.h5")

texts = np.array([], dtype=str) 
times = np.array([], dtype=float) 

when_break = 2
i = 0 
for file in image_files:
       i = i + 1
       if (i > when_break):
              break
       result = extract_and_predict(file, cnn_model=cnn_model, encoder=encoder)
       texts = np.append(texts, result[0])
       times = np.append(times, result[2])
       print(file)
       
print(texts)
print(times)

time_per_letter = []
scores = [] 
letters_u = []
letters_m = []

texts_correct = np.loadtxt('./input/written_name_test_v2.csv', delimiter=",", dtype=str, skiprows=1)
for k in range(0,when_break):
       print(k)
       print(texts_correct[[k]][0][1]) # texts_correct[[i]][0][1] how to get to the correct result 
       score, correct, mistaken = compare_strings(texts_correct[[k]][0][1], texts[k])
       scores.append(score) #score per letter 1/0 values only 
       time_per_letter.append(times[k]/len(texts[k]))
       letters_u.append(correct) 
       letters_m.append(mistaken) 

print("Time: ", np.mean(time_per_letter))
print("Score: ", np.mean(scores))

for word in range(len(letters_u)): 
       print(letters_u[word][0], ' vs ', letters_m[word][0]) 

#print(letters_u)
#print(letters_m)

plt.hist(scores, bins=10, edgecolor='black')
plt.title('Simple Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()












###### PSEUDOCODE ###### 

# for i in images 
##  load image done
###   put it in the model - label binarizer - done
##  save result, time, label - done
## return lists of: results, correct labels, times - done

# accuracy = compare result x label - done
# efficiency  = accuracy/time 

# mean(time, accuracy, efficiency) 

# repeat 1000 times (record the changes? are there any?)\ nope, no need 

############ ############ 


