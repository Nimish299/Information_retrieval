import pickle
import re
import time
import math

excluded_words = set()


def clean_text(text):

    global excluded_words
    
    text = text.lower()

    
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    tokens = text.split()  
    return [token for token in tokens if token not in excluded_words]

def getTokenFrequency(doc_term_frequencies, doc_id, term):
    
    term_freq_list = doc_term_frequencies.get(doc_id, [])
    
    
    left, right = 0, len(term_freq_list) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if term_freq_list[mid][0] == term:
            return term_freq_list[mid][1]
        elif term_freq_list[mid][0] < term:
            left = mid + 1
        else:
            right = mid - 1
    
    
    return 0

def runQuery(doc_ids, doc_term_frequencies, doc_total_tokens, query, collection_frequency,total_tokens):

    ranking = []
    queryTerms = clean_text(query)
    for doc_id in doc_ids:
        score = 1
        for term in queryTerms:
            doc_prob = (getTokenFrequency(doc_term_frequencies, doc_id, term)/doc_total_tokens[doc_id])
            coll_prob = collection_frequency.get(term, 0)/total_tokens
            
            if doc_prob == 0 :
                score *= coll_prob 
            else :
                score *= doc_prob

        ranking.append((doc_id, score))

    ranking.sort(key=lambda x: x[1], reverse=True)
    return ranking

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

def getTermFrequency(inverted_index, term, doc_id):
    
    postings = inverted_index.get(term, {})
    
    
    return postings.get(doc_id, 0.5)

def bm25( doc_ids, query, doc_freq, inverted_inedx, total_tokens, doc_total_tokens) :
    ranking = []
    queryTerms = clean_text(query)
    
    N = len(doc_ids)
    k1 = 1.5 
    b = 0.75
    L_av = total_tokens/N

    for doc_id in doc_ids:
        score = 0
        L_d = doc_total_tokens[doc_id]
        for term in queryTerms:
            df_term = doc_freq.get(term, 0.5)
            tf_term_doc = getTermFrequency(inverted_index, term, doc_id)
            score += (math.log10(N/df_term)) * ( (  (k1 + 1 ) * tf_term_doc ) / (  k1 * ( 1  -  b  +  b *  ( L_d  / L_av ) ) + tf_term_doc  ) )
        ranking.append((doc_id, score))

    ranking.sort(key=lambda x: x[1], reverse=True)
    return ranking

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

    
    with open('scores.pickle', 'rb') as f:
        scores = pickle.load(f)

    
    with open('query_doc_scores.pickle', 'rb') as f:
        query_doc_scores = pickle.load(f)

    
    doc_total_tokens = {}
    doc_term_frequencies = {}
    collection_frequency = {}
    total_tokens = 0
    
    for term, postings in inverted_index.items():
        count = 0
        for doc_id, freq in postings.items():
            total_tokens += freq
            count += freq
            
            if doc_id not in doc_term_frequencies:
                doc_term_frequencies[doc_id] = [(term, freq)]
            else:
                doc_term_frequencies[doc_id].append((term, freq))

            
            if doc_id not in doc_total_tokens:
                doc_total_tokens[doc_id] = freq
            else:
                doc_total_tokens[doc_id] += freq
        collection_frequency[term] = count


    query_id = "PLAIN-1"
    query = "why deep fried foods may cause cancer"
    l = 20
    print("length of scores is : " , l , "\n")

    topdocs = runQuery(doc_ids, doc_term_frequencies, doc_total_tokens, query, collection_frequency, total_tokens)
    print("top 20 for language model : ", topdocs[:20], "\n")
    ndcg = getNdcg(query_id, topdocs[:l], l)
    print("ndcg for language model = ", ndcg, "\n")

    topdocs = bm25(doc_ids, query, doc_freq, inverted_index, total_tokens, doc_total_tokens)
    print("top 20 for bm25 model : ",topdocs[:20], "\n")
    ndcg = getNdcg(query_id, topdocs[:l], l)
    print("ndcg for bm25 = ", ndcg, "\n")


    files = ["nfcorpus/train.nontopic-titles.queries","nfcorpus/train.titles.queries", "nfcorpus/train.vid-desc.queries","nfcorpus/train.vid-titles.queries"]

    for f in files :
        print("------------- file is  : ", f)
        totalLanguageModelTime = 0
        totalBm25ModelTime = 0
        ndcgLanguageModelList = []
        ndcgBm25ModelList = []
        num_queries = 0
        
        with open(f, "r", encoding="utf-8") as file:
            
            for line in file:
                num_queries += 1
                print(num_queries)
                
                parts = line.strip().split("\t")
                
                query_id, query = parts[0], parts[1]

                start_time = time.time()
                topdocs = runQuery(doc_ids, doc_term_frequencies, doc_total_tokens, query, collection_frequency, total_tokens)
                end_time = time.time()
                totalLanguageModelTime += end_time - start_time

                ndcg = getNdcg(query_id, topdocs[:l], l)
                ndcgLanguageModelList.append(ndcg)

                start_time = time.time()
                topdocs = bm25(doc_ids, query, doc_freq, inverted_index, total_tokens, doc_total_tokens)
                end_time = time.time()
                totalBm25ModelTime += end_time - start_time
                
                ndcg = getNdcg(query_id, topdocs[:l], l)
                ndcgBm25ModelList.append(ndcg)

        averageQuerytimeLang = totalLanguageModelTime/num_queries
        averageQuerytimeBm25 = totalBm25ModelTime/num_queries
        averageNdcgLang = getAverageNdcg(ndcgLanguageModelList, num_queries)
        averageNdcgBm25 = getAverageNdcg(ndcgBm25ModelList, num_queries)

        print("averageQuerytimeLang = ",averageQuerytimeLang)
        print("averageQuerytimeBm25 = ",averageQuerytimeBm25)
        print("averageNdcgLang = ", averageNdcgLang)
        print("averageNdcgBm25 = ", averageNdcgBm25)