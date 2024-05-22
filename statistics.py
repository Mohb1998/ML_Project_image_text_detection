import random
import os
import numpy as np
from predict import extract_and_predict 
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

encoder = joblib.load("ML_Project_image_text_detection/label_encoder.pkl")
cnn_model = load_model("ML_Project_image_text_detection/trained_model.h5")

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

def analysis(image_files, encoder, cnn_model, texts_correct, count, words):
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

        if words: 
            correct_word = texts_correct[[k]][0][1]
        else: 
            correct_word = texts_correct
        
        score, correct, mistaken = compare_strings(right = correct_word, model = texts[k])
        scores.append(score) #score per letter 1/0 values only
        lengths_c.append(len(correct_word)) #lenght of the correct word
        lengths_m.append(len(texts[k])) #lenght of the predicted word
        time_per_letter.append(times[k]/lengths_m[k]) #per letter in the predicted word 
        letters_c.append(correct) # what was correct
        letters_m.append(mistaken) # what was the mistake 

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
                
    time_mean_words = np.mean(times)
    time_mean_letter = np.mean(time_per_letter)
    final_correct = sum(scores)
    total_letters = sum(lengths_c)
    total_uncorrect = sum(len(sublist) for sublist in letters_m)
    accuracy = sum(scores)/sum(lengths_c)*100 
    
    return time_mean_words, time_mean_letter, final_correct, total_letters, total_uncorrect, accuracy, pairs, pairs_unique




# LOADING DATA 

#image_dir = "./input/handwritten-characters/Validation/0"

# WHEN USING FOR THE LETTERS 
count  = 100
words = False
main_folder_dir = "./input/handwritten-characters/Validation/" 
main_folder = os.listdir(main_folder_dir) # list of folder names
results = [] # {} into dictionary results['mean'] = mean
for subfolder in main_folder: 
    if subfolder in ["#", "$", "&", "@"]:
            continue
    image_dir = os.path.join(main_folder_dir, subfolder)
    image_files = [os.path.join(image_dir, os.path.normpath(file)) for file in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, file))] 
    texts_correct = subfolder
    time_mean_words, time_mean_letter, final_correct, total_letters, total_uncorrect, accuracy, pairs, pairs_unique = analysis(image_files = image_files, encoder = encoder, cnn_model = cnn_model, texts_correct = texts_correct, count = count, words = words)
    results.append([time_mean_words, time_mean_letter, final_correct, total_letters, total_uncorrect, accuracy, pairs, pairs_unique])


i = 0 
for subfolder in main_folder: 
    if subfolder in ["#", "$", "&", "@"]:
        continue
    print("----- Results for letter ", subfolder, "----------")
    #print("Average time per word:", results[i][0])
    print("Average time per letter:", results[i][1])
    print("Total letters correct:", results[i][2])
    print("Total letters to uncover: ", results[i][3])
    print("Total letters imagined (uncorrect): ", results[i][4])
    print("Accuracy of this model: ", results[i][5], "%")
    print(results[i][6]) #what gets confused the most, with repetition, so that I can do a histogram and see what happens the most 
    print("Unique pairs: ", results[i][7], " out of ", len(results[i][6]))
    print('\n')
    i+=1 


# WHEN USING THE TESTING FOLDER WITH WORDS
#image_dir = "./input/test_v2/test" # test folder 
#count  = 2
#texts_correct = np.loadtxt('./input/written_name_test_v2.csv', delimiter=",", dtype=str, skiprows=1)
#image_files = [os.path.join(image_dir, os.path.normpath(file)) for file in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, file))] 
#print(image_files)
#analysis(image_files = image_files, encoder = encoder, cnn_model = cnn_model, texts_correct = texts_correct, count = count, words = True)

# PLOTTING AREA 
#plt.hist(time_per_letter, bins=10, color='blue')
#plt.title('Simple Histogram - time_per_letter')
#plt.xlabel('Value')
#plt.ylabel('Frequency')
#plt.show()

#plt.hist(pairpairs, bins = pairs_unique, color='blue', edgecolor='black')
#plt.title('Simple Histogram - what gets mistaken the most')
#plt.xlabel('Value')
#plt.ylabel('Frequency')
#plt.show()

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
# - what is confused with what = misrecognition #DONE
# - memory usage, CPU, ... 
# - robustness - ugly images, blurness, light, different backgrounds oh no 
# - different datasets too ? oh no 
