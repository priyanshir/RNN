from io import open
import os
import unicodedata  # needed to covert unicode to ascii
import re
import numpy as np
import torch as t
import torch.nn as nn
import random


class myRNN(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize): #constructor
        super(myRNN, self).__init__()
        self.hiddenSize = hiddenSize
        self.R = nn.Linear(inputSize + hiddenSize, hiddenSize)
        self.O = nn.Linear(inputSize + hiddenSize, outputSize)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hiddenLayer): #forward function
        added = t.cat((input, hiddenLayer), 1)
        hiddenLayer = self.R(added)
        outputLayer = self.O(added)
        outputLayer = self.softmax(outputLayer)
        return outputLayer, hiddenLayer

    def initHidden(self): #hidden layer
        return t.zeros(1, self.hiddenSize)

    def lossFunction(self, predicted, target): #loss Function
        self.loss = nn.CrossEntropyLoss()
        self.output = loss(predicted, target)

    @staticmethod
    def get_list_of_files(directory):  #gets list of file paths in a given directory
        file_paths = os.listdir(directory)
        return file_paths

    @staticmethod
    def print_content(file_path):#prints the content in file path
        file_handle = open(file_path, 'r')
        for line in file_handle:  # goes through each line in the file
            print(line)  # prints each directory file ex. arabic.txt
        file_handle.close()

    @staticmethod
    # https://stackoverflow.com/questions/1207457/convert-a-unicode-string-to-a-string-in-python-containing-extra-symbols credits
    def unicodeTranslate(name):  # translates the accents in the name.
        name = unicodedata.normalize('NFKD', name).encode('ascii','ignore')
        name = ''.join(chr(x) for x in name)
        return name

    @staticmethod
    def returnNamesInFile(file_path):  # returns names in file as a list
        list = []
        input = open(file_path, 'r')
        for line in input:
            names = re.findall(r'([A-Z][^A-Z]+)', line)  # splits the line at the upper case, defines a new name
            for name in names:
                list.append(name)
        return list

    @staticmethod
    def assign_ID(): #assigns an index to every char
        dictionary = {}
        count = 0
        list = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
                "v", "w", "x", "y", "z"]
        for letter in list:
            dictionary[letter] = count
            count += 1
        return dictionary

    @staticmethod
    def turn_letter_into_vector(letter): #turns a char into a vector/tensor
        letter = letter.lower()
        tensor = t.zeros(1, 26)
        dictionary = myRNN.assign_ID()
        tensor[0][dictionary[letter]] = 1
        return tensor

    @staticmethod
    def turn_name_into_vector(name): #turns entire name into a tensor
        tensor = t.zeros(len(name), 1, 26)
        index = 0
        dictionary = myRNN.assign_ID()
        name = myRNN.unicodeTranslate(name)
        for letter in name:
            tensor[index][0][dictionary[letter.lower()]] = 1
            index += 1
        return tensor

    @staticmethod
    def trained_labels(directory):  # just the basic directory with no specified language txt
        list = []
        dictionary = {}
        count = 0
        for country in myRNN.get_list_of_files(directory):
            language = re.search(r'(\w+).txt', country)
            if language:  # if it successfully finds it
                list.append(((language.group(1))))
        for country in list:
            dictionary[country] = count
            count += 1
        return dictionary

    @staticmethod
    def trim(list): #trims the "\n" appearing at the end of the list
        newList = []
        for name in list:
            # print("this is the name: ", name)
            end = name.find("\\")
            name = name[0:end]
            newList.append(name)
        return newList

    @staticmethod
    def splitName(name):
        list = []
        for char in name:
            list.append(char)
        return list

    @staticmethod
    def makeBatch(list_of_input, vocabDict, tagLabels):  # returns the number of batches and the max batch length
        listOfSplittedNames = []
        for name in list_of_input:
            listOfSplittedNames.append(splitName(name))
        max_batch_length = max([len(name) for name in listOfSplittedNames])
        batch_data = vocabDict["PAD"] * np.ones(
            (len(listOfSplittedNames), max_batch_length))  # fill up input with "pad"
        batch_labels = -1 * np.ones((len(listOfSplittedNames), max_batch_length))

        # extract and copy data into batch_data and batch_labels
        for i in range((len(listOfSplittedNames))):
            currentLength = len(listOfSplittedNames[i])
            batch_data[i][:currentLength] = listOfSplittedNames[i]
            batch_labels[i][:currentLength] = tagLabels[i]
        batch_data = t.LongTensor(batch_data)  # new tensor
        batch_labels = t.LongTensor(batch_labels)  # new tensor

    @staticmethod
    def lossFunction(): #another lost function
        return nn.CrossEntropyLoss()

    @staticmethod
    def highestLanguage(output, dictionary): #picks the most likely lanaguage (with the highest probability)
        nation, indexOfLanguage = output.topk(1)
        index = indexOfLanguage[0].item()
        for language in dictionary:
            if (dictionary[language] == index):
                predictedLanguage = language
        return predictedLanguage, index #the language
    @staticmethod
    def trimName(name): #removes the b standing for byte in front of the name
        n = re.search(r'b', name)
        return n
    @staticmethod
    def list_of_countries(countryDict): #just a list of the countries
        list = []
        for country in countryDict:
            list.append(country)
        return list

    @staticmethod
    def language_and_names(): #returns a dictionary with languages as keys and the names in that languages file as the values
        dictionary = {}
        list_of_files = open("C:/Users/priyanshi/Documents/Priyanshi/filenames.txt", 'r')
        listOfFiles = list_of_files.read().split()
        count = 0
        listOfCountries = ["Arabic", "Chinese", "Czech", "Dutch", "English", "French", "German", "Greek", "Irish", "Italian", "Japanese", "Korean", "Polish", "Portuguese", "Russian", "Scottish", "Vietnamese"]
        for line in listOfFiles:
            dictionary[listOfCountries[count]] = myRNN.trim(myRNN.returnNamesInFile(line))
            count += 1
        return dictionary
    @staticmethod
    def randomChoice(choice): #pick a random country/name, choice = list of languages/names
        randomInt = random.randint(0, (len(choice) -1))
        return choice[randomInt]

    @staticmethod
    def trainingEx(countryDict): #train the random example
        list_of_countries = myRNN.list_of_countries(countryDictionary)
        category = myRNN.randomChoice(list_of_countries)
        list_of_names = myRNN.returnNamesInFile("C:/cygwin64/home/priyanshi/rnn_chars/data/data/names/" + (str)(category) + ".txt")
        list_of_names = myRNN.trim(list_of_names)
        languageNames = myRNN.language_and_names() #language with names
        each_line = myRNN.randomChoice(languageNames[category]) #listofnames must be languge: (list of names in that txt)
        category_tensor = t.tensor([list_of_countries.index(category)], dtype=t.long) #filled with many ints
        each_line_tensor = myRNN.turn_name_into_vector(each_line)
        return category, each_line, category_tensor, each_line_tensor
    @staticmethod
    def actualTrainingEx(categoryTensor, lineTensor): #actual training example
        lossFunction = nn.NLLLoss() #because of log softmax
        rnn = myRNN(26, 128, 17) #26 letters in alphabet, 128, and 17 language categories
        hidden = rnn.initHidden() #initializing hidden layer
        for i in range(lineTensor.size()[0]):
            output, hidden = rnn(lineTensor[i], hidden)
        loss = lossFunction(output, categoryTensor)
        loss.backward() #backward function computes the derivatives
        for param in rnn.parameters():
            param.data.add_(-0.005, param.grad.data)
        return output, loss.item()



directory = "C:/cygwin64/home/priyanshi/rnn_chars/data/data/names"
specificDirectory = "C:/cygwin64/home/priyanshi/rnn_chars/data/data/names/Arabic.txt"
print("Files under " + directory + " are:  " + ' '.join(myRNN.get_list_of_files(directory)))
print("Translated version: ", myRNN.unicodeTranslate("SSlusàrski"))
numOfCategories = len(myRNN.trained_labels(directory)) #17
listOfNames = myRNN.trim(myRNN.returnNamesInFile(specificDirectory))

dictionary = myRNN.assign_ID()
countryDictionary = myRNN.trained_labels(directory)
print("the countries: ", countryDictionary)

rnn = myRNN(26, 128, 17)  #the module
input = myRNN.turn_name_into_vector("Max")
myHidden = t.zeros(1, 128) #hidden layer
output, nextHidden = rnn(input[0], myHidden)
print(output) #prints the tensor containing probabilities
print(myRNN.highestLanguage(output, countryDictionary))
for i in range(3): #print 3 different languages with a random name in the languages file
    category, line, category_tensor, line_tensor = myRNN.trainingEx(countryDictionary)
    print('category =', category, '/ line =', line)
currentLoss = 0

category, line, category_tensor, line_tensor = myRNN.trainingEx(countryDictionary)
output, loss = myRNN.actualTrainingEx(category_tensor, line_tensor)
print("OUTPUT", output, loss)
currentLoss += loss #cumlating the loss
guess, guess_i = myRNN.highestLanguage(output, countryDictionary) #guess and guess index
if (guess == category):
    print("CORRECT")
    print(category)
else:
    print("GUESS:", guess, "ACTUAL", category)
    print("WRONG")