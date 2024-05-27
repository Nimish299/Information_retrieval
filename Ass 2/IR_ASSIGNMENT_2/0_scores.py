import pickle


scores = {}
query_doc_scores = {}


with open("nfcorpus/merged.qrel", "r") as file:
    
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


def get_score(query_id, doc_id):
    return scores.get((query_id, doc_id), None)


def get_sorted_doc_scores(query_id):
    return query_doc_scores.get(query_id, [])


print("Score of <PLAIN-1, MED-2421>:", get_score("PLAIN-1", "MED-4070"))

query_id = "PLAIN-1"
print(f"List of pairs of <docid, score> for queryid {query_id}:")
for doc_id, score in get_sorted_doc_scores(query_id):
    print(f"Document ID: {doc_id}, Score: {score}")


with open('scores.pickle', 'wb') as f:
    pickle.dump(scores, f)


with open('query_doc_scores.pickle', 'wb') as f:
    pickle.dump(query_doc_scores, f)