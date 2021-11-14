import pandas as pd
import numpy as np
import os 
import nltk
tokenizer = nltk.RegexpTokenizer(r"\w+")
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer 
ps = PorterStemmer()
from collections import defaultdict
import pickle
import math
from tqdm import tqdm # monitoring progress
import time
from joblib import Parallel, delayed # parallel processing




## Create folders -----------------------------------------------------------------------------------------------------------------------/
def createFolders(nameMainFolder,numberSubFolders):
    for k in range (1, numberSubFolders):
        path = '{}/page_{}'.format(nameMainFolder, k)
        os.makedirs(path)
    
    

## Get htmls by urls -----------------------------------------------------------------------------------------------------------------------/
#these data are useful because they allow us to dinwload more data without seem bot for the server
headers = {
    'user-agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36",
    'accept': "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    'referer': "https://myanimelist.net/"
}

def htmls_by_urls(urls_txt, folder):
     # urls_txt: string 'https.txt' from previous task
    # folder: string; eg '/Users/anton/Desktop/ADM/Homework3/html'
    
    with open(urls_txt, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # list of urls
    list_txt = [line.strip() for line in lines]
    
    i = 0 #through this index you can chose the start point of import
    
    while i < len(list_txt):
        url = list_txt[i]
        # folder where we save html
        al_folder = '{}/page_{}/{}.html'.format(folder, i//50 +1, i+1)
        # download html
        html = requests.get(url, headers)
        print(i)
        if(html.status_code != 200) : 
            time.sleep(120)
            print('error', html.status_code)
        else:
            i += 1
            with open(al_folder, 'w', encoding='utf-8') as g:
                g.write(html.text)

def retriveTSV(folder):
    tsvfile = os.listdir(folder)
    tsvfile = [folder+"/"+ i for i in tsvfile if i.endswith('.tsv')]
    dataset = pd.read_csv(tsvfile[0],sep='\t')
    for tsvid in range(1,len(tsvfile)):
        df1 = pd.read_csv(tsvfile[tsvid],sep='\t')
        dataset = pd.concat([dataset, df1], ignore_index=True)
    return dataset

def getHTML(folder):
    arr = os.listdir(folder)
    alarr = list()
    for y in arr:
        if "rar" in y:
            continue
        fiarr = os.listdir(folder+'/'+y)
        if '.ipynb_checkpoints' in fiarr:
            fiarr.remove('.ipynb_checkpoints')
        for i in range(len(fiarr)):
            fiarr[i] = y+'/'+fiarr[i]
        alarr.extend(fiarr)
    return alarr
            
                
## TOKENIZATION FUNCTION---------------------------------------------------------------------------------------------------------/
def tokenize(description):
    # input: anime description string
    # output: list of tokenized words included in the string
        
    low_descr = str.lower(description)
    
    # We tokenize the description and remove puncuation
    tok_descr = tokenizer.tokenize(low_descr)
    # Alternative way: first tokenize then remove punctuation
    # tok_descr = nltk.word_tokenize(low_descr)
    # nltk.download("punkt")
    # no_pun_descr = [word for word in tok_descr if word.isalnum()]
    
    return tok_descr


## CLEANING FUNCTION---------------------------------------------------------------------------------------------------------/
def clean(tok_descr):
    # input: list of tokenized words included in the string
    # output: list of cleaned words included in the string
    
    # We remove stopwords from tokenized description
    no_stop_descr = [word for word in tok_descr if not word in stopwords.words('english')]
    
    # We carry out stemming
    stem_descr = [ps.stem(i) for i in no_stop_descr]
    
    # We remove isolated characters
    final_descr = [i for i in stem_descr if len(i) > 1]
        
    return final_descr 


## FAST CLEANING FUNCTION---------------------------------------------------------------------------------------------------------/
def clean_fast(tok_descr):
    # Please note: by using intersection of sets instead of list comprehension we lose repeated words within the same description - used to generate dictionaries
    
    # input: list of tokenized words included in the string
    # output: list of cleaned words included in the string
    
    # We remove stopwords from tokenized description
    no_stop_descr = list(set(tok_descr) - (set(tok_descr) & set(stopwords.words('english'))))
    
    # We carry out stemming
    stem_descr = [ps.stem(i) for i in no_stop_descr]
    
    # We remove isolated characters
    final_descr = [i for i in stem_descr if len(i) > 1]
        
    return list(set(final_descr))    


## DICTIONARIES GENERATION---------------------------------------------------------------------------------------------------------/
def dictionaries(dataset):
    # input: anime_df dataframe
    # output 1: the dictionary word_2_id maps word to word identification integer  
    # output 2: the inverted index dictionary id_2_anime maps word identification integer to list of indexes (main dataset indexes) of anime

    word_2_id = defaultdict()
    word_2_id['a'] = 0

    id_2_anime = defaultdict()
        
    for i in tqdm(range(len(dataset))):
        
        final_list = clean_fast(tokenize(dataset['Description'][i]))    
        
        if final_list == []:
            
            pass
        
        else:

            for j in final_list:

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



## SEARCH ENGINE-----------------------------------------------------------------------------------------------------------------------/
def search_engine(query):
    # input: query as string
    # output: list of indexes (anime_df dataframe) of anime whose description contains all the words in the query
    
    # We load dictionaries
    word_2_id_file = open("word_2_id.pkl", "rb")
    word_2_id = pickle.load(word_2_id_file)
    word_2_id_file.close()
    id_2_anime_file = open("id_2_anime.pkl", "rb")
    id_2_anime = pickle.load(id_2_anime_file)
    id_2_anime_file.close()
    
    # We filter query (apply tokenizeandclean function and remove duplicates)
    cleaned_query = list(set(clean(tokenize(query))))
        
    listoflists = []
    
    for i in range(len(cleaned_query)):
        listoflists.append(set(id_2_anime[word_2_id[cleaned_query[i]]]))
        
    anime_intersection = list(set.intersection(*listoflists))
    
    return sorted(anime_intersection)


## Calculate the value TfIdf----------------------------------------------------------------------------------------------------------------------/
def calculate_TfIdf(lenghtDictionary, lenghtTerm, numberOfOccurence, wordsDocument):
    TF = numberOfOccurence / wordsDocument #number of the occurence in the document / #numer of total words in this single document.
    IDF = math.log10(lenghtDictionary / lenghtTerm) #lenght of dictonarty / number of documents that containg the term j
    return round(TF*IDF,2) #just two decimal

## Calculate the number of word occurence in a document-------------------------------------------------------------------------------------------/
def number_occurence(document, word):
    return sum( word in s for s in document) #sum the occurence of a word in a document


## Get details for making a new score -----------------------------------------------------------------------------------------------------------/
# Now we need more information to make the new score
def query_details():
    print("You can leave the input empty if you don't want to spicify it")
    
    print("Write the keywords that anime has")
    query = input()
    
    print("Is it a |TV| series, |movie| or |special|?")
    Type = input()
    print("Roughly how many episodes are there?")
    NumOfEpisodes = input()
    print("How old is the anime? Choose between |new|, |moderate| and |old|")
    AgeOfAnime = input()
    print("How popular is the anime? Choose between |popular|, |moderate| and |not popular|")
    Popularity = input()
    print("If there are any, how many related works (seasons, films, specials, manga) are there?")
    NumOfRelated = input()
    print("You can specify a voice actor if you want")
    Voices = input()
    print("You can specify the name of a staff (producer's name/composer, etc) if you want")
    Staff = input()
    
    return query, Type, NumOfEpisodes, AgeOfAnime, Popularity, NumOfRelated, Voices, Staff


## LIT_EVAL -----------------------------------------------------------------------------------------------------------/
# getting the lists from a string, just a normal ast.literal_eval, but with the expection if there are any
def lit_eval(x):
    try:
        return ast.literal_eval(str(x))   
    except Exception as e:
        return []
    
    
## The main algorithm to calculate the new score ----------------------------------------------------------------------/
def new_score(d, d1, Type, NumOfEpisodes, AgeOfAnime, Popularity, NumOfRelated, Voices, Staff):
    
    #***********************************************************#
    #    d = full dataset                                       #
    #    d1 = query subsection                                  #
    #    Type = ['TV', 'Movie', 'Special']                      #
    #    NumOfEpisodes = n/Episodes                             #
    #    AgeOfAnime = ['new', 'old', 'moderate']                #
    #    Popularity = ['popular', 'moderate', 'not popular']    #
    #    NumOfRelated = ['single', 'few seasons', 'many parts'] #
    #    Voices = [Surname, Name]                               #
    #    Staff = [Surname, Name]                                #
    #***********************************************************#
    
    df = d1.copy()
    
    # Increasing the score if the preferred type is correct
    if len(Type) > 0: #checking if the type is specified
        df.loc[df["Type"] == Type, "NewScore"] += 5
    
    # Increasing the score if the number of episodes is equal or close to the preferred number
    if len(NumOfEpisodes) > 0: #checking if the number of episodes is specified
        NumOfEpisodes = int(NumOfEpisodes) #query was written in str, change to int
        df.loc[np.array(df['Episodes'])-NumOfEpisodes == 0, "NewScore"] += 10 # if the num of episodes match, get the most score
        df.loc[pd.Series(list(abs(np.array(d['Episodes'])-NumOfEpisodes))).between(1, 6)]["NewScore"] += 7 #closer the value, bigger the score
        df.loc[pd.Series(abs(np.array(d['Episodes'])-NumOfEpisodes)).between(7, 12)]["NewScore"] += 4
        df.loc[pd.Series(abs(np.array(d['Episodes'])-NumOfEpisodes)).between(13, 24)]["NewScore"] += 2
        if NumOfEpisodes > 100: #for specifically long animes make this kind of bonus points
            df.loc[np.array(df['Episodes']) >= 100, "Score"] += 10
    
    # Increasing the score if the age of an anime is in the preferred range
    if len(AgeOfAnime) > 0: #checking if the age of an anime is specified
        if AgeOfAnime == "new": 
            # I consider the animes that released after 2015 as "new"
            # but not that old animes get some points too, because you can't be too specific
            df.loc[df['Release date'] > "2015-01-01", "NewScore"] += 5
            df.loc[(df['Release date'] > "2010-01-01") & (df['Release date'] < "2014-12-31"), "NewScore"] += 2
        elif AgeOfAnime == "moderate":
            df.loc[(df['Release date'] > "2008-01-01") & (df['Release date'] < "2014-12-31"), "NewScore"] += 5
            df.loc[(df['Release date'] > "2015-01-01") & (df['Release date'] < "2021-12-31"), "NewScore"] += 2
            df.loc[(df['Release date'] > "2000-01-01") & (df['Release date'] < "2007-12-31"), "NewScore"] += 2
        elif AgeOfAnime == "old":
            # I consider the animes that released before 2000 as "old"
            df.loc[df['Release date'] < "1999-12-31", "NewScore"] += 5
            df.loc[(df['Release date'] > "2000-01-01") & (df['Release date'] < "2007-12-31"), "NewScore"] += 2
        
    # Increasing the score if the popularity of an anime is in the preferred range
    if len(Popularity) > 0: #checking if the popularity is specified
        # of course the popularity depends on the number of people that watched/watching/planning to watch the anime
        if Popularity == "popular": 
            df.loc[df['Members'] > 1000000, "NewScore"] += 10
            df.loc[(df['Members'] > 500000) & (df['Members'] < 999999), "NewScore"] += 8
            df.loc[(df['Members'] > 100000) & (df['Members'] < 499999), "NewScore"] += 5
            df.loc[(df['Members'] > 10000) & (df['Members'] < 99999), "NewScore"] += 2
        elif Popularity == "moderate":
            df.loc[(df['Members'] > 500000) & (df['Members'] < 999999), "NewScore"] += 5
            df.loc[(df['Members'] > 100000) & (df['Members'] < 499999), "NewScore"] += 8
            df.loc[(df['Members'] > 10000) & (df['Members'] < 99999), "NewScore"] += 5
        elif Popularity == "not popular":
            df.loc[df['Members'] < 10000, "NewScore"] += 10
            df.loc[(df['Members'] > 10000) & (df['Members'] < 99999), "NewScore"] += 8
            df.loc[(df['Members'] > 100000) & (df['Members'] < 499999), "NewScore"] += 4
            
    # Increasing the score if the number of related works is close to the preferred number
    if len(NumOfRelated) > 0: #checking if the number of related works is specified
        # Some animes can have sequels, prequels, adaptations, OVAs, specials, etc. 
        # so people can specify how many of them there might be
        NumOfRelated = int(NumOfRelated)
        df['temp'] = df.Related.apply(lambda x: lit_eval(x)) # str to list and store it in temporary column
        df.loc[df['temp'].str.len()-NumOfRelated == 0, "NewScore"] += 6     # Most people don't know the exact number
        df.loc[abs(df['temp'].str.len()-NumOfRelated) < 1, "NewScore"] += 5 # so the score is not scattered that much
        df.loc[abs(df['temp'].str.len()-NumOfRelated) < 3, "NewScore"] += 3
        df.loc[abs(df['temp'].str.len()-NumOfRelated) < 5, "NewScore"] += 1
        # If a person knows that the anime has a lot of related stuff, then we can give it a bigger score
        if NumOfRelated > 15:
            df.loc[df['temp'].str.len() > 15, "NewScore"] += 10
        # delete the temporary column
        del df['temp']
    
    # Increasing the score if the preferred voice actor is in the anime
    if len(Voices) > 0: #checking if the preferred VA is specified
        df['temp'] = df.Voices.apply(lambda x: lit_eval(x)) # str to list and store it in temporary column
        ind = []
        i = 0
        for x in df['temp']: # parsing list of lists and getting the preferred indices
            if Voices in x:
                ind.append(i)
            i += 1
        df.iloc[ind]['NewScore'] += 8
        # delete the temporary column and list
        del df['temp']
        del ind
    
    # Increasing the score if the preferred staff member worked for the anime creation
    if len(Staff) > 0: #checking if the preferred staff member is specified
        df['temp'] = df.Staff.apply(lambda x: lit_eval(x)) # str to list and store it in temporary column
        ind = []
        i = 0
        for x in df['temp']: # parsing list of lists and getting the preferred indices
            if Staff in x:
                ind.append(i)
            i += 1
        df.loc[ind, "NewScore"] += 8
        # delete the temporary column and list
        del df['temp']
        del ind
        
    return df    