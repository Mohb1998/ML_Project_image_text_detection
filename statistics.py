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
#image_dir = "./input/test_v2/test" # test folder 

image_dir = "./input/handwritten-characters/Validation/A"

image_files = [os.path.join(image_dir, os.path.normpath(file)) for file in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, file))] 
encoder = joblib.load("ML_Project_image_text_detection/label_encoder.pkl")
cnn_model = load_model("ML_Project_image_text_detection/trained_model.h5")
texts_correct = np.loadtxt('./input/written_name_test_v2.csv', delimiter=",", dtype=str, skiprows=1)


# MODELLING PART 
count = 200 #number of images to go thru 
texts, times = predict(image_files, count=count, cnn_model = cnn_model, encoder = encoder)

# computing the measurements 
time_per_letter = []
scores = [] 
letters_c = [] #if mistaken this is what was correct
letters_m = [] #if mistaken, this is the mistake 
lengths_c = [] #correct word lenght 
lengths_m = [] #model word lenght 

for k in range(0,count):
       #print(texts_correct[[k]][0][1]) 
       #print(texts[k])
       correct_word = "A" #texts_correct[[k]][0][1]
       score, correct, mistaken = compare_strings(right = correct_word, model = texts[k])
       scores.append(score) #score per letter 1/0 values only
       lengths_c.append(len(correct_word)) #lenght of the correct word
       lengths_m.append(len(texts[k])) #lenght of the predicted word
       time_per_letter.append(times[k]/lengths_m[k]) #per letter in the predicted word 
       letters_c.append(correct) # what was correct
       letters_m.append(mistaken) # what was the mistake 

print("Average time per word:", np.mean(times))
print("Average time per letter:", np.mean(time_per_letter))
print("Total letters correct:", sum(scores))
print("Total letters to uncover: ", sum(lengths_c))
print("Total letters imagined (uncorrect): ", sum(len(sublist) for sublist in letters_m))

print("Accuracy of this model: ", sum(scores)/sum(lengths_c)*100, "%")

pairs = []
pairpairs = []
pairs_unique = 0
for l in range(0, len(letters_c)):
      for m in range(0, len(letters_c[l])):
            if (letters_c[l][m] + letters_m[l][m]) not in pairs:
                pairs_unique = pairs_unique  + 1 
            elif (letters_c[l][m] + letters_m[l][m]) in pairs:
                pairpairs.append(letters_c[l][m] + letters_m[l][m])
            pairs.append(letters_c[l][m] + letters_m[l][m])
            

print(pairs) #what gets confused the most, with repetition, so that I can do a histogram and see what happens the most      
print("Unique pairs: ", pairs_unique, " out of ", len(pairs))


# PLOTTING AREA 
#plt.hist(time_per_letter, bins=10, color='blue')
#plt.title('Simple Histogram - time_per_letter')
#plt.xlabel('Value')
#plt.ylabel('Frequency')
#plt.show()

plt.hist(pairpairs, bins = pairs_unique, color='blue', edgecolor='black')
plt.title('Simple Histogram - what gets mistaken the most')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

#plt.scatter(lengths_c, times, color='blue', marker='o')  
#plt.title('Time dependency, but not correct :) ')
#plt.xlabel('lengths_c')
#plt.ylabel('times')
#plt.show()

#plt.scatter(lengths_m, times, color='blue', marker='o')  
#plt.title('Time dependency')
#plt.xlabel('lengths_m')
#plt.ylabel('times')
#plt.show()


# Saving the figure.
# maybe make it automatic pls, with some organization system pls -> <- #TODO 
#plt.savefig("output.jpg")
 
# Saving figure by changing parameter values
#plt.savefig("output1", bbox_inches="tight", pad_inches=0.3, transparent=True)


# WHAT ELSE CAN WE MEASURE 
# - what is confused with what = misrecognition
# - memory usage, CPU, ... 
# - robustness - ugly images, blurness, light, different backgrounds 
# - different datasets too ? 
