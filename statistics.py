import random
import os
import numpy as np
from predict import extract_and_predict 
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def compare_strings(right, model):
       min_length = min(len(right), len(model))
       score = [] 
       correct = []
       mistaken = []
       for i in range(min_length):
              score.append(right[i] == model[i]) 
              if right[i] != model[i]:
                     correct.append(right[i])
                     mistaken.append(model[i])
       return sum(score), correct, mistaken #sum(score) is the number of letters right in the word 

def predict(image_files, count, cnn_model, encoder): 
    texts = np.array([], dtype=str)   # results from our model 
    times = np.array([], dtype=float) # time it took (whole word)
    i = 0 
    for file in image_files:
        if (i >= count):
            break
        result = extract_and_predict(file, cnn_model = cnn_model, encoder = encoder)
        texts = np.append(texts, result[0])
        times = np.append(times, result[2])
        print(file)
        i += 1
    return texts, times

# LOADING DATA 
image_dir = "./input/test_v2/test" # folder 
image_files = [os.path.join(image_dir, os.path.normpath(file)) for file in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, file))] 
encoder = joblib.load("ML_Project_image_text_detection/label_encoder.pkl")
cnn_model = load_model("ML_Project_image_text_detection/trained_model.h5")
texts_correct = np.loadtxt('./input/written_name_test_v2.csv', delimiter=",", dtype=str, skiprows=1)

# MODELLING PART 
count = 1000 #number of images to go thru 
texts, times = predict(image_files, count=count, cnn_model = cnn_model, encoder = encoder)


# computing the measurements 
time_per_letter = []
scores = [] 
letters_c = []
letters_m = []
lengths_c = [] #correct word lenght 
lengths_m = [] #model word lenght 

for k in range(0,count):
       #print(texts_correct[[k]][0][1]) # texts_correct[[i]][0][1] how to get to the result 
       score, correct, mistaken = compare_strings(right = texts_correct[[k]][0][1], model = texts[k])
       scores.append(score) #score per letter 1/0 values only
       lengths_c.append(len(texts_correct[[k]][0][1]))
       lengths_m.append(len(texts[k]))
       time_per_letter.append(times[k]/lengths_m[k]) #per letter in the predicted word 
       letters_c.append(correct) # what was correct
       letters_m.append(mistaken) # what was the mistake 

print("Average time per word:", np.mean(times))
print("Average time per letter:", np.mean(time_per_letter))
print("Total letters correct:", sum(scores))
print("Total letters to uncover: ", sum(lengths_c))
print("Total letters imagined (uncorrect): ", sum(len(sublist) for sublist in letters_m))



# PLOTTING AREA 
plt.hist(time_per_letter, bins=10, edgecolor='black')
plt.title('Simple Histogram - time_per_letter')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

plt.scatter(lengths_c, times, color='blue', marker='o')  
plt.title('Time dependency, but not correct :) ')
plt.xlabel('lengths_c')
plt.ylabel('times')
plt.show()

plt.scatter(lengths_m, times, color='blue', marker='o')  
plt.title('Time dependency')
plt.xlabel('lengths_m')
plt.ylabel('times')
plt.show()