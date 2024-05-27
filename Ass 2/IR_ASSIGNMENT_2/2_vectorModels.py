
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
        # Convert text to lowercase
        text = text.lower()

        # Remove symbols and number  using regular expressions
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Your text cleaning logic here
        tokens = text.split()  # Example: Tokenization by space
        return [token for token in tokens if token not in excluded_words]

def clean_tokens(tokens_list):
    global excluded_words
    cleaned_tokens_list = []

    for token in tokens_list:
        if token.startswith("http"):
            cleaned_tokens_list.append(token)
            continue  # Skip to the next token

        # Convert token to lowercase
        token = token.lower()

        # Remove symbols and numbers using regular expressions
        token = re.sub(r'[^a-zA-Z]', '', token)

        # Skip token if it's in the excluded words or empty after cleaning
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
    # Create a list of pairs <docId, score>
    doc_scores = [(index_docId_map[i], score) for i, score in enumerate(scores)]
    # Sort the list based on scores in descending order
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    
    return doc_scores, queryVector

def get_score(query_id, doc_id):
    return scores.get((query_id, doc_id), 0)

# Function to get the list of pairs of <docid, score> in descending order given a queryid
def get_sorted_doc_scores(query_id):
    return query_doc_scores.get(query_id, [])

#send only top l docs
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

    # print("ideal scores = ", idealScores, "\n output socres = ", outputScores, "\n")
    # discounting scores
    for i in range (1, l):
        idealScores[i] /= math.log2(i+1)
        outputScores[i] /= math.log2(i+1)

    # summing scores   
    for i in range (1, l):
        idealScores[i] += idealScores[i-1]
        outputScores[i] += outputScores[i-1]
    
    # normalizing scores

    for i in range (0, l):
        ndcg.append(outputScores[i]/idealScores[i])

    return ndcg

def getAverageNdcg(ndcgList, num_queries):
    avg = [0] * len(ndcgList[0])  # Initialize avg with zeros
    
    # Add elements at each index in ndcgList to the corresponding index in avg
    for ndcg in ndcgList:
        avg = [avg[i] + ndcg[i] for i in range(len(ndcg))]
    
    # Divide avg by num_queries to get the average
    avg = [value / num_queries for value in avg]
    
    return avg

if __name__ == "__main__":
    

    with open("nfcorpus/raw/stopwords.large", "r", encoding="utf-8") as file:
        excluded_words.update(map(str.lower, file.read().splitlines()))

    # Load inverted_index
    with open('inverted_index.pickle', 'rb') as f:
        inverted_index = pickle.load(f)

    # Load doc_freq
    with open('doc_freq.pickle', 'rb') as f:
        doc_freq = pickle.load(f)

    # Load doc_ids
    with open('doc_ids.pickle', 'rb') as f:
        doc_ids = pickle.load(f)

    # Load docId_index_map
    with open('docId_index_map.pickle', 'rb') as f:
        docId_index_map = pickle.load(f)

    # Load index_docId_map
    with open('index_docId_map.pickle', 'rb') as f:
        index_docId_map = pickle.load(f)

    # Load scores dictionary
    with open('scores.pickle', 'rb') as f:
        scores = pickle.load(f)

    # Load query_doc_scores dictionary
    with open('query_doc_scores.pickle', 'rb') as f:
        query_doc_scores = pickle.load(f)

    doc_count = len(doc_ids)
    term_count = len(inverted_index)
    start_time = time.time()


    doc_matrix_nnn = np.zeros((doc_count, term_count), dtype=int) # rows = no of docs, cols = no of terms
    doc_matrix_ntn = np.zeros((doc_count, term_count), dtype=int) # rows = no of docs, cols = no of terms

    
    terms = list(inverted_index.keys())

    i = 0
    for term in terms:
        postings = inverted_index.get(term, {})  # Get postings for the current term
        for doc_id, term_freq in postings.items():
            doc_matrix_nnn[docId_index_map[doc_id]][i] = term_freq
            doc_matrix_ntn[docId_index_map[doc_id]][i] = term_freq *  math.log10(doc_count/doc_freq[term])
        i = i + 1

    doc_matrix_ntc = normalize(doc_matrix_ntn, norm='l2', axis=1)

    end_time = time.time()
    print("time for building matrices is " ,end_time - start_time, "\n")


    query_id = "PLAIN-1"
    query = "why deep fried foods may cause cancer"
    l = 20
    print("length of scores is : " , l , "\n")


    start_time = time.time()
    topdocs, queryVector = runQuery(doc_matrix_ntn, terms, query)
    end_time = time.time()
    print("time for running query with ntn ", end_time - start_time, "\n")
    print(topdocs[:20], "\n")
    ndcg = getNdcg(query_id, topdocs[:l], l)
    print("ndcg for ntn = ", ndcg, "\n")

    start_time = time.time()
    topdocs, queryVector = runQuery(doc_matrix_nnn, terms, query)
    end_time = time.time()
    print("time for running query with nnn", end_time - start_time, "\n")
    print(topdocs[:20], "\n")
    ndcg = getNdcg(query_id, topdocs[:l], l)
    print("ndcg for nnn = ", ndcg, "\n")


    start_time = time.time()
    topdocs, queryVector = runQuery(doc_matrix_ntc, terms, query)
    end_time = time.time()
    print("time for running query with ntc", end_time - start_time, "\n")
    print(topdocs[:20], "\n")
    ndcg = getNdcg(query_id, topdocs[:l], l)
    print("ndcg for ntc = ", ndcg, "\n")


    files = ["nfcorpus/train.nontopic-titles.queries","nfcorpus/train.titles.queries", "nfcorpus/train.vid-desc.queries","nfcorpus/train.vid-titles.queries"]


    for f in files :

        print(" ---------------  file is   ", f)
        total_query_time_ntn = 0
        total_query_time_ntc = 0
        total_query_time_nnn = 0
        total_ndcg_ntn = []
        total_ndcg_nnn = []
        total_ndcg_ntc = []

        num_queries = 0

        with open(f, "r", encoding="utf-8") as file:
            for line in file:
                query_id, query = line.strip().split("\t")
                num_queries += 1
                print(num_queries)
                start_time = time.time()
                # Running query with doc_matrix_ntn
                topdocs, queryVector = runQuery(doc_matrix_ntn, terms, query)
                end_time = time.time()
                total_query_time_ntn += end_time - start_time
                ndcg = getNdcg(query_id, topdocs[:l], l)
                total_ndcg_ntn.append(ndcg)

                start_time = time.time()
                # Running query with doc_matrix_nnn
                topdocs, queryVector = runQuery(doc_matrix_nnn, terms, query)
                end_time = time.time()
                total_query_time_nnn += end_time - start_time
                ndcg = getNdcg(query_id, topdocs[:l], l)
                total_ndcg_nnn.append(ndcg)

                start_time = time.time()
                # Running query with doc_matrix_ntc
                topdocs, queryVector = runQuery(doc_matrix_ntc, terms, query)
                end_time = time.time()
                total_query_time_ntc += end_time - start_time
                ndcg = getNdcg(query_id, topdocs[:l], l)
                total_ndcg_ntc.append(ndcg)

        avg_query_time_nnn = total_query_time_nnn / num_queries
        avg_query_time_ntn = total_query_time_ntn / num_queries
        avg_query_time_ntc = total_query_time_ntc / num_queries

        avg_ndcg_ntn = getAverageNdcg(total_ndcg_ntn ,num_queries)
        avg_ndcg_nnn = getAverageNdcg(total_ndcg_nnn ,num_queries)
        avg_ndcg_ntc = getAverageNdcg(total_ndcg_ntc ,num_queries)

        print("Average Query Time (nnn):", avg_query_time_nnn)
        print("Average Query Time (ntn):", avg_query_time_ntn)
        print("Average Query Time (ntc):", avg_query_time_ntc)
        print("Average NDCG (ntn):", avg_ndcg_ntn)
        print("Average NDCG (nnn):", avg_ndcg_nnn)
        print("Average NDCG (ntc):", avg_ndcg_ntc)