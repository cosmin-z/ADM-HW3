import pandas as pd

import nltk
tokenizer = nltk.RegexpTokenizer(r"\w+")
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer 
ps = PorterStemmer()

from collections import defaultdict
import pickle

from tqdm import tqdm



# CLEANING FUNCTION
def tokenizeandclean(description):
    # input: string
    # output: list of filtered words included in the string
    
    # to be applied also to the query
    
    low_descr = str.lower(description)
    
    # We tokenize the description and remove puncuation
    tok_descr = tokenizer.tokenize(low_descr)
    # Alternative way: first tokenize then remove punctuation
    # tok_descr = nltk.word_tokenize(low_descr)
    # nltk.download("punkt")
    # no_pun_descr = [word for word in tok_descr if word.isalnum()]
    
    # We remove stopwords from tokenized description
    no_stop_descr = [word for word in tok_descr if not word in stopwords.words()]
    
    # We carry out stemming
    stem_descr = [ps.stem(i) for i in no_stop_descr]
    
    # We remove isolated characters
    final_descr = [i for i in stem_descr if len(i) > 1]
    
    return final_descr



# DICTIONARIES GENERATION
def dictionaries(dataset):
    # input: anime_df dataframe
    # output 1: the dictionary word_2_id maps word to word identification integer  
    # output 2: the inverted index dictionary id_2_anime maps word identification integer to list of indexes (main dataset indexes) of anime whose cleaned description contains the word identified by the integer

    word_2_id = defaultdict()
    word_2_id['a'] = 0

    id_2_anime = defaultdict()
        
    for i in tqdm(range(len(dataset))):
        
        tok_list = tokenizeandclean(dataset['Description'][i])
        
        if tok_list == []:
            
            pass
        
        else:

            for j in list(set(tok_list)):

                if j not in word_2_id.keys():

                    word_2_id[j] = word_2_id[list(word_2_id.keys())[-1]] + 1

                    id_2_anime[word_2_id[j]] = [i]

                else:

                    id_2_anime[word_2_id[j]].append(i)
    
    # We save dictionaries as pkl
    word_2_id_file = open("word_2_id.pkl", "wb")
    pickle.dump(word_2_id, word_2_id_file)
    word_2_id_file.close()
    
    id_2_anime_file = open("id_2_anime.pkl", "wb")
    pickle.dump(id_2_anime, id_2_anime_file)
    id_2_anime_file.close()

    return word_2_id, id_2_anime



# SEARCH ENGINE
def search_engine(query):
    # input: query as string
    # output: list of indexes (anime_df dataframe) of anime whose description contains all the words in the query
    
    # We load dictionaries
    word_2_id_file = open("dictionaries/word_2_id.pkl", "rb")
    word_2_id = pickle.load(word_2_id_file)
    word_2_id_file.close()
    id_2_anime_file = open("dictionaries/id_2_anime.pkl", "rb")
    id_2_anime = pickle.load(id_2_anime_file)
    id_2_anime_file.close()
    
    # We filter query (apply tokenizeandclean function and remove duplicates)
    cleaned_query = list(set(tokenizeandclean(query)))
        
    listoflists = []
    
    for i in range(len(cleaned_query)):
        listoflists.append(set(id_2_anime[word_2_id[cleaned_query[i]]]))
        
    anime_intersection = list(set.intersection(*listoflists))
    
    return anime_intersection