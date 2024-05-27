def read_query(json_path):
    f = open(json_path + "/s2_query.json", encoding="utf-8")
    json_file = json.load(f)
    
    o = open(json_path + "/intermediate/query.tsv", "w", encoding="utf-8")
    for json_object in json_file['queries']:
        qid = json_object['qid']
        query = json_object['query']
        tokens = query.split(" ")  # Fix the variable name here
        for t in tokens:
            o.write(t.lower() + "\t" + str(qid) + "\n")  
    o.close()