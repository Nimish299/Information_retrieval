import pickle
import re
import time
import math

excluded_words = set()


def clean_text(text):

    global excluded_words
    # Convert text to lowercase
    text = text.lower()

    # Remove symbols and number  using regular expressions
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Your text cleaning logic here
    tokens = text.split()  # Example: Tokenization by space
    return [token for token in tokens if token not in excluded_words]

def getTokenFrequency(doc_term_frequencies, doc_id, term):
    # Get the list of pairs <term, frequency> from doc_term_frequencies for the given doc_id
    term_freq_list = doc_term_frequencies.get(doc_id, [])
    
    # Perform binary search to find the term in the sorted list of pairs
    left, right = 0, len(term_freq_list) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if term_freq_list[mid][0] == term:
            return term_freq_list[mid][1]
        elif term_freq_list[mid][0] < term:
            left = mid + 1
        else:
            right = mid - 1
    
    # If the term is not found, return 0
    return 0

def runQuery(doc_ids, doc_term_frequencies, doc_total_tokens, query, collection_frequency,total_tokens):

    ranking = []
    queryTerms = clean_text(query)
    for doc_id in doc_ids:
        score = 1
        for term in queryTerms:
            doc_prob = (getTokenFrequency(doc_term_frequencies, doc_id, term)/doc_total_tokens[doc_id])
            coll_prob = collection_frequency[term]/total_tokens
            # smoothing
            if doc_prob == 0 :
                score *= coll_prob 
            else :
                score *= doc_prob

        ranking.append((doc_id, score))

    ranking.sort(key=lambda x: x[1], reverse=True)
    return ranking

def get_score(query_id, doc_id):
    return scores.get((query_id, doc_id), 0)

# Function to get the list of pairs of <docid, score> in descending order given a queryid
def get_sorted_doc_scores(query_id):
    return query_doc_scores.get(query_id, [])

#send only top l docs
def getNdcg(query_id, outputRanking, l) :
    idealRanking = get_sorted_doc_scores(query_id)
    idealRanking = idealRanking[:l]
    idealScores = []
    outputScores = []
    ndcg = []
    for p in idealRanking:
        idealScores.append(p[1])

    for p in outputRanking:
        outputScores.append(get_score(query_id, p[0]))

    print("ideal scores = ", idealScores, "\n output socres = ", outputScores, "\n")
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

def getTermFrequency(inverted_index, term, doc_id):
    # Get the postings of the term from the inverted index
    postings = inverted_index.get(term, {})
    
    # Get the frequency for the given document ID from the postings dictionary
    return postings.get(doc_id, 0.5)

def bm25( doc_ids, query, doc_freq, inverted_inedx, total_tokens, doc_total_tokens) :
    ranking = []
    queryTerms = clean_text(query)
    
    N = len(doc_ids)
    k1 = 1.5 # 1.2 to 2
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


if __name__ == "__main__":

    with open("nfcorpus/raw/stopwords.large", "r", encoding="utf-8") as file:
        excluded_words.update(map(str.lower, file.read().splitlines()))

    # Load the inverted index
    with open('inverted_index.pickle', 'rb') as f:
        inverted_index = pickle.load(f)
        
    with open('doc_freq.pickle', 'rb') as f:
        doc_freq = pickle.load(f)

    with open('doc_ids.pickle', 'rb') as f:
        doc_ids = pickle.load(f)

    # Load scores dictionary
    with open('scores.pickle', 'rb') as f:
        scores = pickle.load(f)

    # Load query_doc_scores dictionary
    with open('query_doc_scores.pickle', 'rb') as f:
        query_doc_scores = pickle.load(f)

    # Initialize dictionaries to store the total tokens for each document and terms with frequencies for each document
    doc_total_tokens = {}
    doc_term_frequencies = {}
    collection_frequency = {}
    total_tokens = 0
    # Iterate over the inverted index to populate doc_term_frequencies and doc_total_tokens
    for term, postings in inverted_index.items():
        count = 0
        for doc_id, freq in postings.items():
            total_tokens += freq
            count += freq
            # Update doc_term_frequencies
            if doc_id not in doc_term_frequencies:
                doc_term_frequencies[doc_id] = [(term, freq)]
            else:
                doc_term_frequencies[doc_id].append((term, freq))

            # Update doc_total_tokens
            if doc_id not in doc_total_tokens:
                doc_total_tokens[doc_id] = freq
            else:
                doc_total_tokens[doc_id] += freq
        collection_frequency[term] = count


    with open("test_bm25data1.csv", "w", encoding="utf-8") as output_file:
        with open("nfcorpus/test.titles.queries", "r", encoding="utf-8") as file:
            # Read each line
            for line in file:
                parts = line.strip().split("\t")
                query_id, query = parts[0], parts[1]
                l = len(query_doc_scores[query_id])
                topdocs = bm25(doc_ids, query, doc_freq, inverted_index, total_tokens, doc_total_tokens)
                for doc_id, score in topdocs:
                    total_term_frequency = 0
                    q_tokens = clean_text(query)
                    for t in q_tokens :
                        total_term_frequency += getTermFrequency(inverted_index , t, doc_id) 
                    output_file.write(f"{query_id},{doc_id},{score}\n")

    with open("train_bm25data1.csv", "w", encoding="utf-8") as output_file:  # Change file extension to CSV
        with open("nfcorpus/train.titles.queries", "r", encoding="utf-8") as file:
            # Read each line
            for line in file:
                parts = line.strip().split("\t")
                query_id, query = parts[0], parts[1]
                l = len(query_doc_scores[query_id])
                topdocs = bm25(doc_ids, query, doc_freq, inverted_index, total_tokens, doc_total_tokens)
                for doc_id, score in topdocs:
                    total_term_frequency = 0
                    q_tokens = clean_text(query)
                    for t in q_tokens :
                        total_term_frequency += getTermFrequency(inverted_index , t, doc_id) 
                    output_file.write(f"{query_id},{doc_id},{score}\n")
