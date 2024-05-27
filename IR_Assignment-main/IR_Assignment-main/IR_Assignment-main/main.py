import shutil
import psutil
import os
import json
import numpy as np
import time
import signal
import cProfile
import threading
import statistics
import sys
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from memory_profiler import profile
q3_1_times = []
q3_1_memory = []
q3_2_times = []
q3_2_memory = []
q3_3_times = []
q3_3_memory = []
q_2_times= [] 
q_2_memory= []
q4memory=[]
q5_memory=[]
q41_memory=[]
class TrieNode:
    def __init__(self, reversed_token=""):
        self.children = {}
        self.doc_ids = set()
        self.reversed_token = reversed_token


def read_json_corpus(json_path):
    f = open(os.path.join(json_path, "s2_doc.json"), encoding="utf-8")
    json_file = json.load(f)
    if not os.path.exists(os.path.join(json_path, "intermediate")):
        os.mkdir(os.path.join(json_path, "intermediate"))
    o = open(os.path.join(json_path, "intermediate", "output.tsv"), "w", encoding="utf-8")
    for json_object in json_file['all_papers']:
        doc_no = json_object['docno']
        title = json_object['title'][0]
        paper_abstract = json_object['paperAbstract'][0]
        tokens = title.split(" ")
        for t in tokens:
            o.write(t.lower() + "\t" + str(doc_no) + "\n")
        tokens = paper_abstract.split(" ")
        for t in tokens:
            o.write(t.lower() + "\t" + str(doc_no) + "\n")
    o.close()


def sort(dir):
    f = open(os.path.join(dir, "intermediate", "output.tsv"), encoding="utf-8")
    o = open(os.path.join(dir, "intermediate", "output_sorted.tsv"), "w", encoding="utf-8")


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

def handler(signum, frame):
    raise TimeoutError("Query execution timed out. Memory snapshot:")




def construct_postings(dir, preprocess=False):

    if preprocess:
        o1 = open(dir + "/intermediate/postings_preprocessed.tsv", "w", encoding="utf-8")
    else:
        o1 = open(dir + "/intermediate/postings.tsv", "w", encoding="utf-8")

    postings = {}  
    doc_freq = {}  


    f = open(dir + "/intermediate/output_sorted.tsv", encoding="utf-8")

    sorted_pairs = []


    for line in f:
        line = line[:-1]
        split_line = line.split("\t")
        pairs = (split_line[0], split_line[1])
        sorted_pairs.append(pairs)


    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    for pairs in sorted_pairs:
        token = pairs[0].lower()
        doc_id = pairs[1].lower()

        if preprocess:
            token = preprocess_token(token)


        if not preprocess or (token not in stop_words):
            if token not in postings:
                postings[token] = []
                postings[token].append(doc_id)
            else:
                len_postings = len(postings[token])
                if len_postings >= 1:
                    if doc_id != postings[token][len_postings - 1]:
                        postings[token].append(doc_id)

    for token in postings:
        doc_freq[token] = len(postings[token])

    print("Dictionary size: " + str(len(postings)))


    for token in postings:
        o1.write(token + "\t" + str(doc_freq[token]))
        for doc_id in postings[token]:
            o1.write("\t" + doc_id)
        o1.write("\n")
    o1.close()

    return postings, doc_freq

def load_index_in_memory(dir,pre):
    if pre==0:
        f = open(os.path.join(dir, "intermediate", "postings.tsv"), encoding="utf-8")
    elif pre==1:
         f = open(os.path.join(dir, "intermediate", "postings_preprocessed.tsv"), encoding="utf-8")
    postings = {}
    doc_freq = {}

    for line in f:
        splitline = line.split("\t")

        token = splitline[0]
        freq = int(splitline[1])

        doc_freq[token] = freq

        item_list = []

        for item in range(2, len(splitline)):
            item_list.append(splitline[item].strip())
        postings[token] = item_list
    f.close()
    return postings, doc_freq
def construct_trie(dir):
    trie_root = TrieNode()

   
    f = open(os.path.join(dir, "intermediate/postings.tsv"), encoding="utf-8")


    for line in f:
        line = line[:-1]
        split_line = line.split("\t")
        token, doc_id = split_line[0], split_line[1]

        
        current_node = trie_root
        for char in token:
            if char not in current_node.children:
                current_node.children[char] = TrieNode()
            current_node = current_node.children[char]

      
        current_node.doc_ids.add(doc_id)

    f.close()

    return trie_root


def and_query(query_terms, postings, doc_freq):
   
    postings_for_keywords = {q: postings.get(q, []) for q in query_terms}
    doc_freq_for_keywords = {q: doc_freq.get(q, 0) for q in query_terms}

    sorted_tokens = sorted(doc_freq_for_keywords.items(), key=lambda x: x[1])

    # initialize result to postings list of the token with minimum doc frequency
    result = set(postings_for_keywords[sorted_tokens[0][0]])

    # iterate over the remaining postings list and intersect them with result
    for i in range(1, len(sorted_tokens)):
        result.intersection_update(postings_for_keywords[sorted_tokens[i][0]])
        if not result:
            return result

    return result


def timeout_handler():
    raise TimeoutError("Query execution timed out")

def query_execution_with_timeout(query_terms, postings, doc_freq):
    result = None
    timer = threading.Timer(120, timeout_handler) 

    try:
        timer.start()
        result = and_query(query_terms, postings, doc_freq)
    except TimeoutError:
        pass 
    finally:
        timer.cancel()

    return result
def read_query_and_execute(json_path, index_dir,grep):
    global q3_1_times, q3_1_memory
    start_time_total = time.time()  

    f = open(os.path.join(json_path, "s2_query.json"), encoding="utf-8")
    json_file = json.load(f)


    results_file = open(os.path.join(json_path, "intermediate", "query_results_Q3.1.tsv"), "w", encoding="utf-8")

 
    query_times = []
    postings, doc_freq = load_index_in_memory(index_dir, 0)

    for json_object in json_file['queries']:

        start_time_query = time.time()

        qid = json_object['qid']
        query = json_object['query']


        query_terms = query.split()
        boolean_query = ' and '.join(query_terms)

       
        profiler = cProfile.Profile()
        try:
            profiler.enable()
            if use_grep:
                result = execute_query_with_grep(boolean_query)
            else:
                result = query_execution_with_timeout(query_terms, postings, doc_freq)
        finally:
            query_time = time.time() - start_time_query
            profiler.disable()

        if result is not None:
            query_times.append(query_time)

          
            results_file.write(f"{qid}\t{boolean_query.lower()}\t{query_time}\t{len(result)}\n")



    results_file.close()

    total_time = time.time() - start_time_total
    print("Experiment 1:")
    print("\nTotal Time for All Queries using Boolean Retrieval :", total_time)

    if query_times:
        print("Maximum Query Time:", max(query_times))
        print("Minimum Query Time:", min(query_times))
        print("Average Query Time:", statistics.mean(query_times))

    q3_1_times.append(total_time)
    q3_1_memory.append(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)

def preprocess_token(token):

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    processed_token = lemmatizer.lemmatize(stemmer.stem(token))
    return processed_token



def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


def create_preprocessed_query_file(json_path):
    f = open(os.path.join(json_path, "s2_query.json"), encoding="utf-8")
    json_file = json.load(f)
    f.close()

    
    preprocessed_query_file = open(os.path.join(json_path, "intermediate", "preprocessed_query.tsv"), "w", encoding="utf-8")

    for json_object in json_file['queries']:
        qid = json_object['qid']
        query = json_object['query']

     
        processed_query_terms = preprocess_text(query)
        preprocessed_query = ' '.join(processed_query_terms)

        preprocessed_query_file.write(f"{qid}\t{preprocessed_query}\n")


    preprocessed_query_file.close()


def read_and_execute_boolean_queries(json_path, index_dir):
    global q3_2_times, q3_2_memory
    start_time_total = time.time()  

    f = open(os.path.join(json_path, "intermediate", "preprocessed_query.tsv"), encoding="utf-8")
    preprocessed_queries = [line.strip() for line in f]
    f.close()


    results_file = open(os.path.join(json_path, "intermediate", "query_results_Q3.2.tsv"), "w", encoding="utf-8")


    query_times = []
    postings, doc_freq = load_index_in_memory(index_dir, 1)

    for preprocessed_query_line in preprocessed_queries:
        start_time_query = time.time()  # Start timer for each query

        qid, preprocessed_query = preprocessed_query_line.split('\t')

   
        processed_query_terms = preprocessed_query.split()
        boolean_query = ' and '.join(processed_query_terms)

        profiler = cProfile.Profile()
        try:
            profiler.enable()
            result = query_execution_with_timeout(processed_query_terms, postings, doc_freq)
        finally:
            query_time = time.time() - start_time_query
            profiler.disable()

        if result is not None:
            query_times.append(query_time)

            results_file.write(f"{qid}\t{boolean_query.lower()}\t{query_time}\t{len(result)}\n")

    results_file.close()

    total_time = time.time() - start_time_total
    print("Experiment 2:")
    print("\nTotal Time for All Queries using post-processing of the vocabulary:", total_time)

    if query_times:
        print("Maximum Query Time:", max(query_times))
        print("Minimum Query Time:", min(query_times))
        print("Average Query Time:", statistics.mean(query_times))

    q3_2_times.append(total_time)
    q3_2_memory.append(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)


def collect_docs_with_prefix(node):
    result = set()

    if node.doc_ids:
        result.update(node.doc_ids)

    for child_node in node.children.values():
        result.update(collect_docs_with_prefix(child_node))

    return result

def and_query_trie(query_terms, trie_root):
    result = None
   
    
    for term in query_terms:
      
        current_node = trie_root

        for char in term:
            if char not in current_node.children:
                break
            current_node = current_node.children[char]

        term_doc_ids = collect_docs_with_prefix(current_node)

        if result is None:
            result = term_doc_ids.copy()
        else:
   
            result.intersection_update(term_doc_ids)
            if not result:
                break

  

    return result


def query_execution_with_timeout_and_query_trie(query_terms, trie_root):
    result = None
    timer = threading.Timer(120, timeout_handler)  
    try:
        timer.start()
        result = and_query_trie(query_terms, trie_root)
    except TimeoutError:
        pass 
    finally:
        timer.cancel()

    return result

def read_query_and_execute_trie(json_path, trie_root):
    global q3_3_times, q3_3_memory
    start_time_total = time.time() 

    f = open(os.path.join(json_path, "s2_query.json"), encoding="utf-8")
    json_file = json.load(f)


    results_file = open(os.path.join(json_path, "intermediate", "query_results_trie_q3.tsv"), "w", encoding="utf-8")


    query_times = []

    for json_object in json_file['queries']:
        start_time_query = time.time() 

        qid = json_object['qid']
        query = json_object['query']

        query_terms = query.split()

        profiler = cProfile.Profile()
        try:
            profiler.enable()
            result = query_execution_with_timeout_and_query_trie(query_terms, trie_root)
        except TimeoutError:
            pass  
        finally:
            profiler.disable()

        if result is not None:
            query_time = time.time() - start_time_query
            query_times.append(query_time)

            results_file.write(f"{qid}\t{query}\t{query_time}\t{len(result)}\n")


    results_file.close()

    total_time = time.time() - start_time_total
    print("Experiment 3:")
    print("\nTotal Time for All Queries tree-based implementation of dictionaries :", total_time)

    if query_times:
        print("Maximum Query Time:", max(query_times))
        print("Minimum Query Time:", min(query_times))
        print("Average Query Time:", statistics.mean(query_times))

    q3_3_times.append(total_time)
    q3_3_memory.append(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)  # in MB

def compare_query_times_and_memory(json_path):
  
    print("Experiment 3.2:")

   
    print("Total Query Time:", max(q3_1_times))
    print("Maximum Memory Usage:", max(q3_1_memory))

   
    print("Total Query Time:", max(q3_2_times))
    print("Maximum Memory Usage:", max(q3_2_memory))

  
    print("Total Query Time:", max(q3_3_times))
    print("Maximum Memory Usage:", max(q3_3_memory))


class PermutermIndex:
    def __init__(self, base_directory):
        self.postings_file_path = os.path.join(base_directory, "intermediate", "postings.tsv")
        self.index = {}
        terms = self.read_terms_from_postings(self.postings_file_path)
        self.construct_permuterm_indexes(terms)

    def rotate_term(self, term):
        """Rotate the given term and return a list of rotations."""
        rotations = [term[i:] + term[:i] for i in range(len(term))]
        return rotations

    def construct_permuterm_indexes(self, terms):
        """Construct Permuterm indexes for the given terms."""
        for term in terms:
            rotations = self.rotate_term(term + '$')  
            for rotation in rotations:
                self.index[rotation] = term

    def prefix_search(self, query):
        """Perform a prefix search using the Permuterm index."""
        query += '$'
        for i in range(len(query)):
            if query[i] == '*':
                rotated_query = query[i+1:] + query[:i+1]  # Rotate the query until '*' appears at the end
                matching_terms = [value for key, value in self.index.items() if key.startswith(rotated_query[:-1])]
                return matching_terms
     
        print("Error: Wildcard (*) not found in the query.")
        return []

    def read_terms_from_postings(self, postings_file_path):
        """Read terms from the postings file and return a list."""
        try:
            with open(postings_file_path, encoding="utf-8") as file:
                terms = [line.split("\t")[0] for line in file]
            return terms
        except FileNotFoundError:
            print(f"Error: File '{postings_file_path}' not found.")
            return []

    def write_indexes_to_file(self, output_file_path):
        """Write Permuterm indexes to a file."""
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write("RotatedTerm\tOriginalTerm\n")
            for rotated_term, original_term in self.index.items():
                output_file.write(f"{rotated_term}\t{original_term}\n")


def benchmark_permuterm_queries(permuterm_index, base_dir):
    global q41_memory
    output_file_path = os.path.join(base_dir, "intermediate", "wildcard_benchmark.tsv")
    file = open(os.path.join(base_dir, "s2_wildcard.json"), 'r')
    queries = json.load(file)['queries']
    file.close()

    query_times = []
    total_memory_usage = 0.0
    output_file = open(output_file_path, 'w', encoding='utf-8')
    output_file.write("QueryID\tWildcardQuery\tQueryTime\tNumResults\n")

    for query_obj in queries:
        query_id = query_obj['qid']
        wildcard_query = query_obj['query']

        start_time = time.time()
        results = permuterm_index.prefix_search(wildcard_query)
        query_time = time.time() - start_time
        query_times.append(query_time)

        output_file.write(f"{query_id}\t{wildcard_query}\t{query_time:.6f}\t{len(results)}\n")
 
        current_memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        total_memory_usage += current_memory_usage


    output_file.close()
    q41_memory.append(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)
    print("Experiment 4.1:")

    if query_times:
        print("\nQuery Time Statistics:")
        print("Minimum Query Time:", min(query_times))
        print("Maximum Query Time:", max(query_times))
        print("Average Query Time:", statistics.mean(query_times))
        print(f"\nTotal Memory Usage: ",q41_memory)
    else:
        print("\nNo query times recorded.")



def construct_Reverse_trie(dir):
    trie_root = TrieNode()

    f = open(os.path.join(dir, "intermediate/postings.tsv"), encoding="utf-8")

    for line in f:
        line = line[:-1]
        split_line = line.split("\t")
        token, doc_id = split_line[0], split_line[1]

        reversed_token = token[::-1]

        current_node = trie_root
        for char in reversed_token:
            if char not in current_node.children:
                current_node.children[char] = TrieNode(reversed_token)
            current_node = current_node.children[char]

        current_node.doc_ids.add(doc_id)

    f.close()

    return trie_root



class TreeIndex:
    def __init__(self, forward_trie_root, backward_trie_root):
        self.forward_trie_root = forward_trie_root
        self.backward_trie_root = backward_trie_root

    def forward_search(self, query):
        results = self.search_in_trie(self.forward_trie_root, query)
        return results

    def backward_search(self, query):
        results = self.search_in_trie(self.backward_trie_root, query)
        return results

    def wildcard_query(self, query):
        if '*' not in query:
           forward_results = self.forward_search(query + '*')
           return list(forward_results)
        parts = query.split('*')
        if len(parts) != 2:
            print(f"Invalid wildcard query: {query}")
            return []

        forward_query = parts[0]
        backward_query = parts[1]
        

        forward_results = self.forward_search(forward_query + '*')

        backward_results = self.backward_search(backward_query[::-1] + '*')
        intersection_results = set(forward_results).intersection(backward_results)
        return list(intersection_results)

    def search_in_trie(self, trie_root, query):
        current_node = trie_root
        for char in query:
            if char == '*':
                return self.collect_docs_with_prefix(current_node)
            elif char in current_node.children:
                current_node = current_node.children[char]
            else:
                return []
        return list(current_node.doc_ids)

    def collect_docs_with_prefix(self, node):
        result = set(node.doc_ids)
        for child_node in node.children.values():
            result.update(self.collect_docs_with_prefix(child_node))
        return result

def benchmark_tree_based_queries(tree_index, base_dir):
    global q4memory
    output_file_path = os.path.join(base_dir, "intermediate", "wildcard_tree_benchmark.tsv")
    file = open(os.path.join(base_dir, "s2_wildcard.json"), 'r')
    queries = json.load(file)['queries']
    file.close()
 
    query_times = []

    output_file = open(output_file_path, 'w', encoding='utf-8')
    output_file.write("QueryID\tWildcardQuery\tQueryTime\tNumResults\n")

    profiler = cProfile.Profile()
    profiler.enable()

    for query_obj in queries:
        query_id = query_obj['qid']
        wildcard_query = query_obj['query']

        start_time = time.time()
        results = tree_index.wildcard_query(wildcard_query)
        query_time = time.time() - start_time
        query_times.append(query_time)

        output_file.write(f"{query_id}\t{wildcard_query}\t{query_time:.6f}\t{len(results)}\n")
  
    profiler.disable()


    output_file.close()
    q4memory.append(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)
    print("Experiment 4.2:")
    if query_times:
        print("\nWildcard Query Time Statistics:")
        print("Minimum Query Time:", min(query_times))
        print("Maximum Query Time:", max(query_times))
        print("Average Query Time:", statistics.mean(query_times))
        print(f"\nTotal Memory Usage:", q4memory)
    else:
        print("\nNo wildcard query times recorded.")

def read_and_execute_mixed_boolean_queries(json_path, tree_index):
    global q5_memory
    start_time_total = time.time()

    f = open(os.path.join(json_path, "s2_wildcard_boolean.json"), encoding="utf-8")
    wildcard_queries = json.load(f)['queries']
    f.close()

    results_file = open(os.path.join(json_path, "intermediate", "query_results_mixed_boolean.tsv"), "w", encoding="utf-8")

    query_times_list = []

    profiler = cProfile.Profile()
    profiler.enable()

    for query_obj in wildcard_queries:
        start_time_query = time.time()

        qid = query_obj['qid']
        query = query_obj['query']

        query_terms = query.split()

        results_for_terms = []
        for term in query_terms:
            term_result = tree_index.wildcard_query(term)
            results_for_terms.append(term_result)
       
        result = set.intersection(*map(set, results_for_terms))
        query_time = time.time() - start_time_query
        query_times_list.append(query_time)

        results_file.write(f"{qid}\t{query.lower()}\t{query_time}\t{len(result)}\n")

    profiler.disable()


    results_file.close()
    print("Experiment 5:")
    total_time = time.time() - start_time_total
    print("\nTotal Time for All Queries:", total_time)
    q5_memory.append(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)
    if query_times_list:
        print("Maximum Query Time:", max(query_times_list))
        print("Minimum Query Time:", min(query_times_list))
        print("Average Query Time:", statistics.mean(query_times_list))
        print(f"\nTotal Memory Usage:", q5_memory)
    

def index(dir): 
    intermediate_dir = os.path.join(dir, "intermediate")
    print("Choose an option:")
    print("1. Execute boolean queries with inverted index")
    print("2. Preprocessed vocabulary and Execute boolean queries with inverted index")
    print("3. Execute boolean queries with trie")
    print("4. Compare hash with tree")
    print("5. Benchmark Permuterm Index")
    print("6. Benchmark Tree-based Index")
    print("7. Execute mixed boolean queries")
    print("8. ALL")
    print("9. Exit")

    while True:
        choice = input("Enter your choice (1-9): ")

        if choice == '1':
            read_json_corpus(dir)
            sort(dir)
            construct_postings(dir, preprocess=False)
            read_query_and_execute(dir, dir,0)  # 3.1
        elif choice == '2':
            create_preprocessed_query_file(dir)
            construct_postings(dir, preprocess=True)
            read_and_execute_boolean_queries(dir, dir)  # 3.2
        elif choice == '3':
            trie_root = construct_trie(dir)
            read_query_and_execute_trie(dir, trie_root)  # 3.3
        elif choice == '4':
            compare_query_times_and_memory(dir)
        elif choice == '5':
            permuterm_index = PermutermIndex(dir)
            # size_of_index = sys.getsizeof(permuterm_index)
            print(f"Size of Permuterm Index: {size_of_index} bytes")
            permuterm_index_output_file_path = os.path.join(intermediate_dir, "permuterm_indexes.tsv")
            permuterm_index.write_indexes_to_file(permuterm_index_output_file_path)
            benchmark_permuterm_queries(permuterm_index, dir)
        elif choice == '6':
            trie_root = construct_trie(dir)
            trie_root_rev = construct_Reverse_trie(dir)
            tree_index = TreeIndex(trie_root, trie_root_rev)
            size_of_index = sys.getsizeof(tree_index)
            # print(f"Size of Tree-based Index: {size_of_index} bytes")
            benchmark_tree_based_queries(tree_index, dir)
        elif choice == '7':
            read_and_execute_mixed_boolean_queries(dir, tree_index)
        elif choice == '8':
                intermediate_dir = os.path.join(dir, "intermediate")
                read_json_corpus(dir)
                sort(dir)
                construct_postings(dir, preprocess=False)   
                read_query_and_execute(dir, dir) #3.1
                create_preprocessed_query_file(dir)
                construct_postings(dir, preprocess=True)
                read_and_execute_boolean_queries(dir, dir)#3.2
                trie_root = construct_trie(dir)
                read_query_and_execute_trie(dir, trie_root)#3.3
                compare_query_times_and_memory(dir)
                permuterm_index = PermutermIndex(dir)
                size_of_index = sys.getsizeof(permuterm_index)
                print(f"Size of Permuterm Index: {size_of_index} bytes")
                permuterm_index_output_file_path = os.path.join(intermediate_dir, "permuterm_indexes.tsv")
                permuterm_index.write_indexes_to_file(permuterm_index_output_file_path)
                benchmark_permuterm_queries( permuterm_index,dir)
                trie_root_rev = construct_Reverse_trie(dir)
                tree_index = TreeIndex(trie_root, trie_root_rev)
                size_of_index = sys.getsizeof(tree_index)
                print(f"Size of Tree-based Index: {size_of_index} bytes")
                benchmark_tree_based_queries(tree_index, dir)
                read_and_execute_mixed_boolean_queries(dir, tree_index)
        elif choice == '9':
            read_query_and_execute(dir, dir,1)  
            
        else:
            print("Invalid choice. Please enter a number between 1 and 8.")
            break


if __name__ == '__main__':
    index('s2/')
