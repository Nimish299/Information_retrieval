from ahocorasick import Automaton
import csv
import numpy as np
import pickle
import math
import re
import time
excluded_words = set()


class TrieNode:
    def __init__(self):
        
        self.children = {}
        self.output = []
        self.fail = None
        self.tf = 0  

def parse_gen_data(gen_data_path):
    entities = set()
    with open(gen_data_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  
        for row in reader:
            subject = row[0]
            object_ = row[2]
            entities.add(subject)
            entities.add(object_)
    return entities
def parse_gen_data_subject(gen_data_path):
    entities = set()
    with open(gen_data_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  
        for row in reader:
            subject = row[0]
            
            entities.add(subject)
            
    return entities
def parse_gen_data_object(gen_data_path):
    entities = set()
    with open(gen_data_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  
        for row in reader:
            
            object_ = row[2]
            
            entities.add(object_)
    return entities
def build_automaton(keywords):
    
    root = TrieNode()

    
    for keyword in keywords:
        node = root
        
        for char in keyword:
            node = node.children.setdefault(char, TrieNode())
        
        node.output.append(keyword)

    
    queue = []
    
    for node in root.children.values():
        queue.append(node)
        node.fail = root

    
    while queue:
        current_node = queue.pop(0)
        
        for key, next_node in current_node.children.items():
            queue.append(next_node)
            fail_node = current_node.fail
            
            while fail_node and key not in fail_node.children:
                fail_node = fail_node.fail
            
            next_node.fail = fail_node.children[key] if fail_node else root
            
            next_node.output += next_node.fail.output

    return root


def search_text(text, keywords):
    
    root = build_automaton(keywords)
    
    result = {keyword: 0 for keyword in keywords}

    current_node = root
    
    for i, char in enumerate(text):
        
        while current_node and char not in current_node.children:
            current_node = current_node.fail

        if not current_node:
            current_node = root
            continue

        
        current_node = current_node.children[char]
        
        for keyword in current_node.output:
            result[keyword] += 1

    return result

def index(entities, filePath) :
    i = 0
    doc_matrix = []
    with open(filePath , "r", encoding='utf-8') as file:
        for line in file:
            doc_vector = []
            fields = line.split()
            doc_id = fields[0]
            result = search_text(line, entities)
            for pattern, tf in result.items():
                
                doc_vector.append(tf)
            doc_matrix.append(doc_vector)
            i = i + 1

    return doc_matrix

def multiply(array_2d, array_1d) :

    array_1d_column = array_1d[:, np.newaxis]
    result = np.dot(array_2d, array_1d_column)
    return result.flatten().tolist()


def runQueryUpdated(doc_matrix_nnn, queryVector) :

    scores = multiply(doc_matrix_nnn, queryVector)
    
    doc_scores = [(index_docId_map[i], score) for i, score in enumerate(scores)]
    
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    
    return doc_scores

def get_score(query_id, doc_id):
    return scores.get((query_id, doc_id), 0)


def get_sorted_doc_scores(query_id):
    return query_doc_scores.get(query_id, [])


def getAverageNdcg(ndcgList, num_queries):
    avg = [0] * len(ndcgList[0])  
    
    
    for ndcg in ndcgList:
        avg = [avg[i] + ndcg[i] for i in range(len(ndcg))]
    
    
    avg = [value / num_queries for value in avg]
    
    return avg


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



gena_data_path = 'genadata/gena_data_final_triples1.csv'
filePath = "nfcorpus/raw/doc_dump.txt"
entities = parse_gen_data(gena_data_path)
entities1 = parse_gen_data_subject(gena_data_path)
entities2 = parse_gen_data_object(gena_data_path)
print(entities)


with open('index_docId_map.pickle', 'rb') as f:
    index_docId_map = pickle.load(f)
with open('scores.pickle', 'rb') as f:
        scores = pickle.load(f)

    
with open('query_doc_scores.pickle', 'rb') as f:
        query_doc_scores = pickle.load(f)

doc_matrix = index(entities, filePath)



query_vector = []

query = "Flibanserin NEW00001 is a CHEMICAL Flibanserin, a multifunctional serotonin receptor agonist and antagonist, is currently approved in the United States and Canada for the treatment of acquired, generalized hypoactive sexual desire disorder (HSDD) in premenopausal women"
query_id = "PLAIN-1"


result = search_text(query, entities)
for pattern, tf in result.items():
    x = 0
    if tf > 0:
        x = 1
    query_vector.append(x)

l = 20
print(query_vector)

print(doc_matrix)
print(query_vector)
topdocs = runQueryUpdated(np.array(doc_matrix), np.array(query_vector))
print(topdocs[:10])
ndcg = getNdcg(query_id, topdocs[:l], l)
print(ndcg[:5])


files = ["nfcorpus/train.nontopic-titles.queries","nfcorpus/train.titles.queries", "nfcorpus/train.vid-desc.queries","nfcorpus/train.vid-titles.queries"]

for f in files:
        print(" ----------------- file is : ", f)
        queryTimeTotal = 0
        ndcgList = []
        num_queries = 0
        
        with open(f, "r", encoding="utf-8") as file:
            
            for line in file:
                num_queries += 1
                
                
                parts = line.strip().split("\t")
                
                query_id, query = parts[0], parts[1]
                
                start_time = time.time()
                query_vector = []
                result = search_text(query, entities)
                for pattern, tf in result.items():
                        x = 0
                        if tf > 0:
                            x = 1
                        query_vector.append(x)
                
                topdocs = runQueryUpdated(np.array(doc_matrix), np.array(query_vector))
                end_time = time.time()
                queryTimeTotal += end_time-start_time 
                
                ndcg = getNdcg(query_id, topdocs[:l], l)
                ndcgList.append(ndcg)
        
        averageQueryTime = queryTimeTotal/num_queries
        averageNdcg = getAverageNdcg(ndcgList, num_queries)

        print("average Query time =  ", averageQueryTime)
        print("average ndcg = ", averageNdcg)


