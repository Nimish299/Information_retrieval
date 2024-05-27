
import re
import time
import numpy as np
import math
from sklearn.preprocessing import normalize
import pickle

excluded_words = set()
doc_ids = []
docId_index_map = {}
index_docId_map = {}


def clean_text(text):

        global excluded_words
        
        text = text.lower()

        
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        tokens = text.split()  
        return [token for token in tokens if token not in excluded_words]

def clean_tokens(tokens_list):
    global excluded_words
    cleaned_tokens_list = []

    for token in tokens_list:
        if token.startswith("http"):
            cleaned_tokens_list.append(token)
            continue  

        
        token = token.lower()

        
        token = re.sub(r'[^a-zA-Z]', '', token)

        
        if token.lower() not in excluded_words and token.strip():
            cleaned_tokens_list.append(token)

    return cleaned_tokens_list


def getIndex(arr, target):
    
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
            
    return -1

def getQueryVector(terms , query):
    queryTerms = clean_text(query)
    queryVector = np.zeros(len(terms), dtype=int)
    for term in queryTerms:
        index = getIndex(terms, term)
        if(index != -1):
            queryVector[index] = 1
    
    return queryVector

def multiply(array_2d, array_1d) :

    array_1d_column = array_1d[:, np.newaxis]
    result = np.dot(array_2d, array_1d_column)
    return result.flatten().tolist()

def runQuery(doc_matrix_nnn, terms, query) :
    queryVector = getQueryVector(terms, query)
    scores = multiply(doc_matrix_nnn, queryVector)
    
    doc_scores = [(index_docId_map[i], score) for i, score in enumerate(scores)]
    
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    
    return doc_scores, queryVector

def runQueryUpdated(doc_matrix_nnn, terms, queryVector) :

    scores = multiply(doc_matrix_nnn, queryVector)
    
    doc_scores = [(index_docId_map[i], score) for i, score in enumerate(scores)]
    
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    
    return doc_scores

def get_score(query_id, doc_id):
    return scores.get((query_id, doc_id), 0)


def get_sorted_doc_scores(query_id):
    return query_doc_scores.get(query_id, [])


def getNdcg(query_id, outputRanking, l) :
    idealRanking = get_sorted_doc_scores(query_id)

    idealScores = []
    outputScores = []
    ndcg = []
    for p in idealRanking:
        idealScores.append(p[1])

    idealScores = idealScores[:l] + [0] * max(l - len(idealScores), 0)

    for p in outputRanking:
        outputScores.append(get_score(query_id, p[0]))

    
    

    
    for i in range (1, l):
        idealScores[i] /= math.log2(i+1)
        outputScores[i] /= math.log2(i+1)

    
    for i in range (1, l):
        idealScores[i] += idealScores[i-1]
        outputScores[i] += outputScores[i-1]
    
    
    for i in range (0, l):
        ndcg.append(outputScores[i]/idealScores[i])

    return ndcg

def updateQueryVector(doc_matrix_ntc, topdocs, docId_index_map) :
    relavent_docs = topdocs[:25]
    irrelavent_docs = topdocs[25:]

    rel_centroid = np.zeros(term_count)
    irr_centroid = np.zeros(term_count)

    for p in relavent_docs:
        rel_centroid = rel_centroid + doc_matrix_ntc[docId_index_map[p[0]]]

    rel_centroid = rel_centroid/25

    for p in irrelavent_docs:
        irr_centroid = irr_centroid + doc_matrix_ntc[docId_index_map[p[0]]]

    irr_centroid = irr_centroid/len(irrelavent_docs)

    alpha = 1
    beta = 0.75
    gamma = 0.25

    updated_query_vector = alpha * queryVector + beta * rel_centroid - gamma * irr_centroid
    updated_query_vector = np.maximum(updated_query_vector, 0)
    return updated_query_vector

def getAverageNdcg(ndcgList, num_queries):
    avg = [0] * len(ndcgList[0])  
    
    
    for ndcg in ndcgList:
        avg = [avg[i] + ndcg[i] for i in range(len(ndcg))]
    
    
    avg = [value / num_queries for value in avg]
    
    return avg

if __name__ == "__main__":
    
    with open("nfcorpus/raw/stopwords.large", "r", encoding="utf-8") as file:
        excluded_words.update(map(str.lower, file.read().splitlines()))

    
    with open('inverted_index.pickle', 'rb') as f:
        inverted_index = pickle.load(f)

    
    with open('doc_freq.pickle', 'rb') as f:
        doc_freq = pickle.load(f)

    
    with open('doc_ids.pickle', 'rb') as f:
        doc_ids = pickle.load(f)

    
    with open('docId_index_map.pickle', 'rb') as f:
        docId_index_map = pickle.load(f)

    
    with open('index_docId_map.pickle', 'rb') as f:
        index_docId_map = pickle.load(f)

    
    with open('scores.pickle', 'rb') as f:
        scores = pickle.load(f)

    
    with open('query_doc_scores.pickle', 'rb') as f:
        query_doc_scores = pickle.load(f)



    doc_count = len(doc_ids)
    term_count = len(inverted_index)
    start_time = time.time()


    doc_matrix_ntn = np.zeros((doc_count, term_count), dtype=int) 

    
    terms = list(inverted_index.keys())

    i = 0
    for term in terms:
        postings = inverted_index.get(term, {})  
        for doc_id, term_freq in postings.items():
            doc_matrix_ntn[docId_index_map[doc_id]][i] = term_freq *  math.log10(doc_count/doc_freq[term])
        i = i + 1

    doc_matrix_ntc = normalize(doc_matrix_ntn, norm='l2', axis=1)

    end_time = time.time()
    print("time for building matrix is " ,end_time - start_time, "\n")


    query_id = "PLAIN-1"
    query = "why deep fried foods may cause cancer"
    l = 20
    print("length of scores is : " , l , "\n")
    
    start_time = time.time()
    topdocs, queryVector = runQuery(doc_matrix_ntc, terms, query)
    print(queryVector)
    end_time = time.time()
    print("time for running query ", end_time - start_time, "\n")
    print("topdocs initially", topdocs[:20], "\n")
    ndcg = getNdcg(query_id, topdocs[:l], l)
    print("ndcg  = ", ndcg, "\n")

    updated_query_vector = updateQueryVector(doc_matrix_ntc, topdocs, docId_index_map)
    print(updated_query_vector)
    topdocs = runQueryUpdated(doc_matrix_ntc, terms, updated_query_vector)
    print("topdocs after query update ", topdocs[:20], "\n")

    ndcg = getNdcg(query_id, topdocs[:l], l)
    print("ndcg  after query update = ", ndcg, "\n")


    files = ["nfcorpus/train.nontopic-titles.queries","nfcorpus/train.titles.queries", "nfcorpus/train.vid-desc.queries","nfcorpus/train.vid-titles.queries"]

    for f in files:
        print(" ----------------- file is : ", f)
        queryTimeTotal = 0
        updateTimeTotal = 0
        ndcgListBefore = []
        ndcgListAfter = []
        num_queries = 0

        
        with open(f, "r", encoding="utf-8") as file:
            
            for line in file:
                num_queries += 1
                print(num_queries)
                
                parts = line.strip().split("\t")
                
                query_id, query = parts[0], parts[1]

                start_time = time.time()
                topdocs, queryVector = runQuery(doc_matrix_ntc, terms, query)
                end_time = time.time()

                queryTimeTotal += end_time-start_time 

                ndcg = getNdcg(query_id, topdocs[:l], l)
                ndcgListBefore.append(ndcg)

                start_time = time.time()
                updated_query_vector = updateQueryVector(doc_matrix_ntc, topdocs, docId_index_map)
                end_time = time.time()

                updateTimeTotal += end_time-start_time
                
                start_time = time.time()
                topdocs = runQueryUpdated(doc_matrix_ntc, terms, updated_query_vector)
                end_time = time.time()

                queryTimeTotal += end_time-start_time 

                ndcg = getNdcg(query_id, topdocs[:l], l)
                ndcgListAfter.append(ndcg)
        
        averageQueryTime = queryTimeTotal/num_queries
        averageUpdateTime = updateTimeTotal/num_queries
        averageNdcgBefore = getAverageNdcg(ndcgListBefore, num_queries)
        averageNdcgAfter = getAverageNdcg(ndcgListAfter, num_queries)

        print("average Query time =  ", averageQueryTime)
        print("average update time = ", averageUpdateTime)
        print("average ndcg before = ", averageNdcgBefore)
        print("average ndcg after = ", averageNdcgAfter)