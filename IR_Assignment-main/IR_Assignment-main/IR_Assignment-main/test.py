import os
import subprocess
import json

# Rest of your code...

def execute_query_with_grep(query, index_file_path):
    grep_command = f"grep -i -E '{query}' {index_file_path}"

    try:
        process = subprocess.run(grep_command, shell=True, capture_output=True, text=True, check=True)
        result = process.stdout
    except subprocess.CalledProcessError as e:
        result = None

    return result

def index(dir):
    index_file_path = os.path.join(dir, 's2_doc.json')  # Use os.path.join to handle path concatenation
    search_term = 'deep learning'
    result = execute_query_with_grep(search_term, index_file_path)  
    print(result)

# Code starts here
if __name__ == '__main__':
    index('s2/')
