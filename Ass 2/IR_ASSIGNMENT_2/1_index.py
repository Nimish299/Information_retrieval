import re
import time
import pickle
import sys
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

        
        if token not in excluded_words and token.strip():
            cleaned_tokens_list.append(token)

    return cleaned_tokens_list

def read_documents(path1):
    o = open('output.tsv', "w", encoding="utf-8")
    i = 0
    with open(path1 , "r", encoding='utf-8') as file:
        for line in file:
            fields = line.split()
            doc_id = fields[0]
            doc_ids.append(doc_id)
            docId_index_map[doc_id] = i
            index_docId_map[i] = doc_id
            tokens = clean_tokens(fields[1:])
            for t in tokens:
                o.write(t.lower() + "\t" + doc_id + "\n")
            i = i + 1
    o.close()

def sort():
    f = open('output.tsv', encoding="utf-8")
    o = open('output_sorted.tsv', "w", encoding="utf-8")

    pairs = []

    for line in f:
        line = line[:-1]
        split_line = line.split("\t")
        if len(split_line) == 2:
            pair = (split_line[0], split_line[1])
            pairs.append(pair)

    sorted_pairs = sorted(pairs, key=lambda x: (x[0], x[1]))

    for sp in sorted_pairs:
        o.write(sp[0] + "\t" + sp[1] + "\n")
    o.close()

def construct_postings():
    o1 = open('postings.tsv', "w", encoding="utf-8")

    postings = {}  
    doc_freq = {}  

    f = open('output_sorted.tsv', encoding="utf-8")

    for line in f:
        line = line[:-1]
        split_line = line.split("\t")
        term = split_line[0]
        doc_id = split_line[1]

        if term not in postings:
            postings[term] = {}
            postings[term][doc_id] = 1
        else:
            if doc_id in postings[term]:
                postings[term][doc_id] += 1
            else:
                postings[term][doc_id] = 1

    for term in postings:
        doc_freq[term] = len(postings[term])

    for term in postings:
        o1.write(term + "\t" + str(doc_freq[term]))
        for doc_id, term_freq in postings[term].items():
            o1.write("\t" + doc_id + ":" + str(term_freq))
        o1.write("\n")
    o1.close()

def load_index_in_memory():
    f = open('postings.tsv', encoding="utf-8")
    postings = {}
    doc_freq = {}

    for line in f:
        split_line = line.split("\t")

        term = split_line[0]
        freq = int(split_line[1])

        doc_freq[term] = freq

        postings[term] = {}

        for item in split_line[2:]:
            doc_id, term_freq = item.split(":")
            postings[term][doc_id] = int(term_freq)

    return postings, doc_freq

def index(path1):
    read_documents(path1)
    sort()
    construct_postings()
    return load_index_in_memory()

def process_qrel_file(path):
    scores = {}
    query_doc_scores = {}

    with open(path, "r") as file:
        for line in file:
            parts = line.strip().split("\t")
            query_id, _, doc_id, score = parts
            
            scores[(query_id, doc_id)] = int(score)
            
            if query_id not in query_doc_scores:
                query_doc_scores[query_id] = [(doc_id, int(score))]
            else:
                query_doc_scores[query_id].append((doc_id, int(score)))

    for query_id in query_doc_scores:
        query_doc_scores[query_id].sort(key=lambda x: x[1], reverse=True)

    return scores, query_doc_scores

def get_score(query_id, doc_id):
    return scores.get((query_id, doc_id), 0)


def get_sorted_doc_scores(query_id):
    return query_doc_scores.get(query_id, [])
    
if __name__ == "__main__":
    path1 = "nfcorpus/raw/doc_dump.txt"
    path_qrel = "nfcorpus/merged.qrel"

    with open("nfcorpus/raw/stopwords.large", "r", encoding="utf-8") as file:
        excluded_words.update(map(str.lower, file.read().splitlines()))

    
    start_time = time.time()
    inverted_index, doc_freq = index(path1)
    print(len(inverted_index))
    with open('inverted_index.pickle', 'wb') as f:
        pickle.dump(inverted_index, f)

    with open('doc_freq.pickle', 'wb') as f:
        pickle.dump(doc_freq, f)

    with open('doc_ids.pickle', 'wb') as f:
        pickle.dump(doc_ids, f)

    with open('docId_index_map.pickle', 'wb') as f:
        pickle.dump(docId_index_map, f)

    with open('index_docId_map.pickle', 'wb') as f:
        pickle.dump(index_docId_map, f)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time for indexing:", elapsed_time, "seconds")

    
    scores, query_doc_scores = process_qrel_file(path_qrel)
    
    
    with open('scores.pickle', 'wb') as f:
        pickle.dump(scores, f)

    
    with open('query_doc_scores.pickle', 'wb') as f:
        pickle.dump(query_doc_scores, f)
    

    
    
    
    


    
    
    