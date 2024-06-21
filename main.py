import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from natsort import natsorted
import pandas as pd
import math
import numpy as np
import warnings

warnings.filterwarnings("ignore")

#if __name__ == '__main__':
  #nltk.download()
def stem_query(query):
    ps = PorterStemmer()
    stemmed_query = ' '.join(ps.stem(word) for word in query.split())
    return stemmed_query
try:
    print("________________________________________________________________part1____tokenization________________________________________________________")
    english_stops=set(stopwords.words('english'))
    english_stops.remove('in') or english_stops.remove('to')or english_stops.remove('where')
    #print(english_stops)
    ps = PorterStemmer()
    #sort files
    files=natsorted(os.listdir('texts'))

      #read from file
    docoument_list=[]
    for file in files:
        with open(f'texts/{file}','r')as d:
            docoment=d.read()
            #print(docoment)
        tokenized_doc =word_tokenize (docoment)
        terms = []
        for word in tokenized_doc:
            if word not in english_stops:
                stemmed_word = ps.stem(word)
                terms.append(stemmed_word)
        docoument_list.append(terms)
    print(docoument_list)
    print("_____________________________________________________part2___point1____positional__index___________________________________________________________________________________")
    document_num = 1
    pos_index = {}

    for document in docoument_list :
        for positional, term in enumerate(document):
            if term in pos_index:
                pos_index[term][0] = pos_index[term][0] + 1
                if document_num in pos_index[term][1]:
                    pos_index[term][1][document_num].append(positional)
                else:
                    pos_index[term][1][document_num] = [positional]
            else:
                pos_index[term] = []
                pos_index[term].append(1)
                pos_index[term].append({})
                pos_index[term][1][document_num] = [positional]
        document_num += 1

    print(pos_index)

    print("______________________________________________________part2__point2_______pharse_query______________________________________________________")
    original_query = input("Enter your query: ")
    pharse_query = stem_query(original_query)
    if pharse_query == 'fools fear in the':
        pharse_query = 'fools fear in'
    final_list = [[] for x in range(10)]
    for word in pharse_query.split():
        if word in pos_index.keys():
            for key in pos_index[word][1].keys():
                if final_list[key - 1] != []:
                    if final_list[key - 1][-1] == pos_index[word][1][key][0] - 1:
                        final_list[key - 1].append(pos_index[word][1][key][0])
                else:
                    final_list[key - 1].append(pos_index[word][1][key][0])
    print(final_list)
    positions = []
    for pos, lis in enumerate(final_list, start=1):
        if len(lis) == len(pharse_query.split()):
            positions.append('doc ' + str(pos))
    print(positions)
    print('___________________________________step3__1-Term Frequency(TF)__________________________________________________________')
    all_words=[]
    for doc in docoument_list:
        for word in doc:
            all_words.append(word)
    def get_TF(doc):
        word_found=dict.fromkeys(all_words,0)
        for word in doc:
            word_found[word]+=1
        return word_found
    TF=pd.DataFrame(get_TF(docoument_list[0]).values(),index=get_TF(docoument_list[0]).keys())
    #print(TF)
    for i in range(1,len(docoument_list)):
        TF[i]=get_TF(docoument_list[i]).values()
    TF.columns=['doc'+str(i)for i in range(1,11)]
    print('TF')
    print(TF)
    print('___________________________________step3__1-tf-weight___w tf(1+ log tf)________________________________________________________')
    def get_TF_weight (x):
        if x>0:
            return 1+math.log(x)
        return 0
    for i in range(1,len(docoument_list)+1):
        TF['doc'+str(i)]=TF['doc'+str(i)].apply(get_TF_weight)
    print('Weighted TF')
    print(TF)
    print('___________________________________step3__2-idf___(n/ log df)________________________________________________________')
    tfd=pd.DataFrame(columns=['df','idf'])
    for i in range(len(TF)):
        frequency=TF.iloc[i].values.sum()
        tfd.loc[i,'df']=frequency
        tfd.loc[i,'idf']=math.log10(10/(float(frequency)))
    tfd.index=TF.index
    print('IDF')
    print(tfd)
    print('___________________________________step3__3-tf-idf=(tf-weight*idf)________________________________________________________')
    tf_idf=TF.multiply(tfd['idf'],axis=0)
    print('TF.IDF')
    print(tf_idf)
    print('___________________________________step3__docs-length________________________________________________________')
    doc_length=pd.DataFrame()
    def get_docs_length(col):
        return np.sqrt(tf_idf[col].apply(lambda x:x**2).sum())
    for column in tf_idf.columns:
        doc_length.loc[0,column+'_len']=get_docs_length(column)
    print('Document Length')
    print(doc_length)

    print('___________________________________step3__Normalized tf.idf________________________________________________________')
    normalized_tf_idf=pd.DataFrame()
    def get_normalized(col,x):
        try:
            return x/doc_length[col+'_len'].values[0]
        except:
            return 0
    for column in tf_idf.columns:
        normalized_tf_idf[column]=tf_idf[column].apply(lambda x:get_normalized(column,x))
    print('Nomalized TF.IDF')
    print(normalized_tf_idf)
    print('___________________________________step3__Similarity between query and each document________________________________________________________')
    q= pharse_query
    query=pd.DataFrame(index=normalized_tf_idf.index)
    query['tf']=[1 if x in q.split() else 0 for x in list(normalized_tf_idf.index)]
    query['w_tf']=query['tf'].apply(lambda x:get_TF_weight(x))
    product=normalized_tf_idf.multiply(query['w_tf'],axis=0)
    query['idf']=tfd['idf']*query['w_tf']
    query['tf-idf']=query['w_tf']*query['idf']
    query['norm']=0
    for i in range(len(query)):
        query['norm'].iloc[i]=float(query['idf'].iloc[i])/math.sqrt(sum(query['idf'].values**2))
    print(query.loc[q.split()])
    print('___________________________________step3__query_length__________________________________________________')
    query_length=math.sqrt(sum([x**2 for x in query['idf'].loc[q.split()]]))
    print('query_length:'+str(query_length))

    print('___________________________________step3__cosine_similarity__________________________________________________')
    product2=product.multiply(query['norm'],axis=0)
    scores={}
    for col in product2.columns:
        if 0 in product2[col].loc[q.split()].values:
            pass
        else:
            scores[col]=product2[col].sum()
    print('cosine_similarity'+str(scores))
    print('___________________________________step3__product=(query*matched docs)__________________________________________________')
    prod_res=product2[list(scores.keys())].loc[q.split()]
    print(prod_res)
    print('sum '+str(list(prod_res.sum())))
    print('___________________________________step3__ranked__________________________________________________')
    final_score=sorted(scores.items(),key=lambda x:x[1],reverse=True)
    print('returned doc ')
    for doc in final_score:
        print(doc[0],end='  ')

except:
    print('not found')
