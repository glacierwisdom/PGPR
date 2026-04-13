import requests
import json
from urllib.parse import quote

def test_query(ensp_id):
    clean_id = ensp_id.split('.')[-1] if '.' in ensp_id else ensp_id
    
    queries = [
        clean_id,
        f'xref:ensembl-{clean_id}',
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    results = {}
    for q in queries:
        # Test with and without fields
        urls = [
            f"https://rest.uniprot.org/uniprotkb/search?query={quote(q)}&format=json",
            f"https://rest.uniprot.org/uniprotkb/search?query={quote(q)}&format=json&fields=accession,protein_name,cc_function"
        ]
        for url in urls:
            try:
                print(f"Testing URL: {url}")
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    results_list = data.get('results', [])
                    num_results = len(results_list)
                    print(f"  Success! Found {num_results} results.")
                    if num_results > 0:
                        print(f"  Sample result: {json.dumps(results_list[0], indent=2)}")
                else:
                    print(f"  Failed with status code: {response.status_code}")
            except Exception as e:
                print(f"  Error: {e}")
    
    return results

if __name__ == "__main__":
    test_id = "P01308"
    test_query(test_id)
