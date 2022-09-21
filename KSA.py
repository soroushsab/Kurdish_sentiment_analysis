#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np




# # A function to remove useless characters

# In[2]:


def remove_bad_characters(allText):
    allText = allText.replace('!',' ')
    allText = allText.replace('.!',' ')
    allText = allText.replace('!!!',' ')
    allText = allText.replace('%',' ')
    allText = allText.replace('"',' ')
    allText = allText.replace(',',' ')
    allText = allText.replace('،',' ')
    allText = allText.replace('(',' ')
    allText = allText.replace(')',' ')
    allText = allText.replace(']',' ')
    allText = allText.replace('[',' ')
    allText = allText.replace('}',' ')
    allText = allText.replace('{',' ')
    allText = allText.replace(':',' ')
    allText = allText.replace(';',' ')
    allText = allText.replace('.',' ')
    allText = allText.replace('????','?')
    allText = allText.replace('???','?')
    allText = allText.replace('??','?')
    allText = allText.replace('*',' ')
    allText = allText.replace('/',' ')
    allText = allText.replace('//',' ')
    allText = allText.replace('  ',' ')
    allText = allText.replace('   ',' ')
    allText = allText.replace('\\n_\\n',' ')
    return allText


# # A function to read stop words

# In[3]:


def StopWords():
    tempLits = []
    with open('untitled.txt', 'r',encoding='utf-8') as f:
        for row in f:
            row = row[:-1]
            tempLits.append(row)
    f.close()
    return tempLits
stopwords = StopWords()


# # A function to read all comments and return two Lists
# ## neg_comments would be negative comments (the file was temp/done_neg2.txt)
# ## pos_comments would be positive comments (the file was temp/done_pos2.txt)

# In[4]:


def readComments():
    negativeComments = []
    with open('temp/done_neg2.txt', 'r',encoding='utf-8') as f:
        for row in f:
            r = remove_bad_characters(row[:-1])
            r = r.split(" ")
            negativeComments.append(r)
    f.close()
    positiveComments = []
    with open('temp/done_pos2.txt', 'r',encoding='utf-8') as f:
        for row in f:
            r = remove_bad_characters(row[:-1])
            r = r.split(" ")
            positiveComments.append(r)
    f.close()
    return negativeComments, positiveComments
neg_comments , pos_comments = readComments()


# # A function to do normalizeation 
# ## remove stop words and useless words

# In[5]:


def normalize(List,polarity):
    tempList = []
    for el in List:
        temp1 = []
        for word in el:
            if word not in stopwords and (not word.isdigit()) and len(word)>1:
                temp1.append(word)
        if len(temp1) > 1:
            temp1.append(polarity)
            tempList.append(temp1)
    return tempList


# # get the tokenize data

# In[6]:


tokenized_pos_comments = normalize(pos_comments,1)
tokenized_neg_comments = normalize(neg_comments,-1)


# # get the list of all words

# In[7]:


def list_of_all_terms(poses,negs):
    List_of_all_words = []
    temp = poses+negs
    for el in temp:
        for word in el[:-1]:
            if word not in List_of_all_words:
                List_of_all_words.append(word)
    return List_of_all_words
    
List_of_all_words = list_of_all_terms(tokenized_pos_comments,tokenized_neg_comments)


# In[9]:


all_comments =tokenized_pos_comments+tokenized_neg_comments


# In[13]:


# some statistics
print(len(List_of_all_words))
print(len(all_comments))


# In[20]:


def calcualte_TFIDF(all_comments):
    import math
    TF = {}
    TTF = {}
    DF = {}
    i = 0
    for doc in all_comments:
        seen_list = []
        temp_tf = {}
        for word in doc[:-1]:
            if word not in seen_list:
                if word in DF:
                    DF[word] += 1
                else:
                    DF[word] = 1
                temp_tf[word] = doc.count(word)
                seen_list.append(word)
            if word in TTF:
                TTF[word] += doc.count(word)
            else:
                TTF[word] = doc.count(word)
        temp_tf['class'] = doc[-1]
        TF[i] = temp_tf
        i+=1
    TF_IDF = {}
    i = 0
    for el in TF:
        temp_tfidf = {}
        for term in List_of_all_words:
            if term in TF[el] and term in TTF and term in DF:
                temp_tfidf[term] = (TF[el][term] / TTF[term]) * ((math.log(len(all_comments)/(1+DF[term]))))
            else:
                temp_tfidf[term] = 0
        temp_tfidf['class'] = TF[el]['class']
        TF_IDF[i] = temp_tfidf
        i+=1
    return TF_IDF


# In[21]:


res = calcualte_TFIDF(all_comments)


# In[34]:


list_for_dataset = []
for el in res:
    list_for_dataset.append(list(res[el].values()))


# In[ ]:





# In[37]:


# with open('Final_2022.txt', 'w',encoding='utf-8') as f:
#     for line in list(res[0].keys()):
#         f.write(str(line) + ', ')
#     f.write('\n')
#     for line in list_for_dataset:
#         for wo in line:
#             f.write(str(wo) + ', ')
#         f.write('\n')
# f.close()


# In[ ]:


df = pd.DataFrame(list_for_dataset, columns=list(res[0].keys()))

