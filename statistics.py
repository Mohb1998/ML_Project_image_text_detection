import random
import os
import numpy as np
from predict import extract_and_predict 
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

encoder = joblib.load("ML_Project_image_text_detection/old_trained_models/700, 300, 7 epochs, acc 38,05/label_encoder.pkl")
cnn_model = load_model("ML_Project_image_text_detection/old_trained_models/700, 300, 7 epochs, acc 38,05/trained_model.h5")

def compare_strings(right, model):
       min_length = min(len(right), len(model))
       score = [] 
       correct = []
       mistaken = []
       for i in range(min_length):
              if right in [' ', "#", "$", "&", "@", "-"]:  #skipping white space & symbols 
                   continue
              if (model[i] == '0'): #for evaluating the words only, not letters
                   continue
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
    #print("Len texts: ", len(texts))

    # computing the measurements    
    time_per_letter = []
    scores = [] 
    letters_c = [] #if mistaken this is what was correct    
    letters_m = [] #if mistaken, this is the mistake 
    lengths_c = [] #correct word lenght 
    lengths_m = [] #model word lenght 
    score_words = 0

    for k in range(0,min(count,len(texts))):
        #print(texts_correct[[k]][0][1]) 
        #print("k is ", k, ", result is " ,texts[k])
        #print("K",k)
        #print(len(texts_correct))
        if words: 
            l = k 
            file_name = image_files[l].split("\\")[1] #the image itself
            while texts_correct[[l]][0][0] != file_name:
                if l == len(texts_correct): 
                    print("image not found")
                    break
                else: 
                    l = l+1 
            correct_word = texts_correct[[l]][0][1]    
                #if texts_correct[[l]][0][0] == file_name:
                    #print("gotcha")

        else: 
            correct_word = texts_correct

        print("Correct word:", correct_word)
        print("Estimated word:", texts[k])
        
        
        score, correct, mistaken = compare_strings(right = correct_word, model = texts[k])
        if (correct_word == texts[k] or len(mistaken)<=2):
            score_words = score_words + 1
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

    results = {   
        'word': correct_word,
        'correct_words': score_words,
        'time_mean_words': np.mean(times), 
        'time_mean_letter' : np.mean(time_per_letter), 
        'final_correct' : sum(scores), 
        'total_letters' : sum(lengths_c), 
        'total_uncorrect' : sum(len(sublist) for sublist in letters_m), 
        'accuracy' : sum(scores)/sum(lengths_c)*100,
        'pairs' : pairs,
        'pairpairs': pairpairs,
        'pairs_unique' : pairs_unique
    }
    return results



# LOADING DATA 

# WHEN ANALYSING JUST THE LETTERS
'''
count  = 100 #the maximum number of pictures it can take in a folder 
words = False
main_folder_dir = "./input/handwritten-characters/Validation/" 
main_folder = os.listdir(main_folder_dir)  #list of folder names
results = [] 
letters = []
for subfolder in main_folder: 
    if subfolder in ["#", "$", "&", "@", '1', '2', '3', '4', '5', '6', '7', '8', '9']: #ignore chars and numbers
            continue
    image_dir = os.path.join(main_folder_dir, subfolder)
    image_files = [os.path.join(image_dir, os.path.normpath(file)) for file in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, file))] #list of images inside a folder
    texts_correct = subfolder #just for some clarity 
    results.append(analysis(image_files = image_files, encoder = encoder, cnn_model = cnn_model, texts_correct = texts_correct, count = count, words = words))
    letters.append(subfolder)


accuracies = [] 
times_per_letter = []
for i in range(0, len(results)): 
    print("----- Results for letter ", results[i]['word'], "----------")
    #print("Average time per word:", results[i]['time_mean_words'])
    print("Average time per letter:", results[i]['time_mean_letter'])
    print("Total letters correct:", results[i]['final_correct'])
    print("Total letters to uncover: ", results[i]['total_letters'])
    print("Total letters imagined (uncorrect): ", results[i]['total_uncorrect'])
    print("Accuracy of this model: ", results[i]['accuracy'], "%")
    print(results[i]['pairs']) #what gets confused the most, with repetition, so that I can do a histogram and see what happens the most 
    print("Unique pairs: ", results[i]['pairs_unique'], " out of ", len(results[i]['pairs']))
    print('\n')
    accuracies.append(results[i]['accuracy'])
    times_per_letter.append(results[i]['time_mean_letter'])
    if results[i]['pairs_unique'] > 1:  
        plt.hist(results[i]['pairs'], bins = results[i]['pairs_unique'], color='blue', edgecolor='black')
        plt.title(f'Confusion histogram - character {results[i]['word']}')
        plt.xlabel('character pairs')
        plt.ylabel('frequency')
        plt.savefig(f'./ML_Project_image_text_detection/results/conf_hist-{results[i]['word']}-n={results[i]['total_letters']}.png', bbox_inches="tight")
        plt.show()


print("Average accuracy: ", np.mean(accuracies), "%")
print(letters)
print(accuracies)    

plt.bar(letters, accuracies, color='blue', edgecolor='black')
plt.title('Accuracy per character')
plt.xlabel(' ')
plt.ylabel('accuracy in %')
plt.savefig(f'./ML_Project_image_text_detection/results/Accuracy-chars-n=max.png', bbox_inches="tight")
plt.show()

plt.bar(letters, times_per_letter, color='blue', edgecolor='black')
plt.title('Average time per character')
plt.xlabel(' ')
plt.ylabel('average time in ms')
plt.savefig(f'./ML_Project_image_text_detection/results/Average-time-chars-n=max.png', bbox_inches="tight")
plt.show()
'''

# WHEN ANALYSING WHOLE WORDS 

image_dir = "./input/test_subset" # test folder 
count  = 50
results = [] 
texts_correct = np.loadtxt('./input/written_name_test_v2.csv', delimiter=",", dtype=str, skiprows=1)
image_files = [os.path.join(image_dir, os.path.normpath(file)) for file in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, file))] 
#print(image_files)
results = (analysis(image_files = image_files, encoder = encoder, cnn_model = cnn_model, texts_correct = texts_correct, count = count, words = True))
    
accuracies = [] 
correct_words = []
#print(results)
times_per_letter = []
print("----- Results for words, n =", count, "----------")
print("Average time per word:", results['time_mean_words'])
print("Average time per letter:", results['time_mean_letter'])
print("Total words correct:", results['correct_words'])
print("Total letters correct:", results['final_correct'])
print("Total letters to uncover: ", results['total_letters'])
print("Total letters imagined (incorrect): ", results['total_uncorrect'])
print("Accuracy of this model: ", results['accuracy'], "%")
#print(results['pairs']) #what gets confused the most, with repetition, so that we can do a histogram and see what happens the most 
print("Unique pairs: ", results['pairs_unique'], " out of ", len(results['pairs']))
print('\n')


plt.hist(results['pairpairs'], bins = results['pairs_unique'], color='blue', edgecolor='black') #pairspairs is pairs, but those that repeat more then once, to make the hist readable
plt.title(f'Confusion histogram - words, n = {count}')
plt.xlabel('character pairs')
plt.ylabel('frequency - 1')
plt.savefig(f'./ML_Project_image_text_detection/results/conf_hist-n={results['total_letters']}.png', bbox_inches="tight")
plt.show()

