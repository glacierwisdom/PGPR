import pandas as pd
import requests
import time
import logging
from pathlib import Path
from tqdm import tqdm
from urllib.parse import quote

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_protein_info(ensp_id):
    """
    Fetch protein name and function from UniProt using Ensembl Protein ID.
    """
    # Remove taxon ID prefix if present (e.g., 9606.ENSP... -> ENSP...)
    clean_id = ensp_id.split('.')[-1] if '.' in ensp_id else ensp_id
    
    # Use the clean ID directly as the query, it's the most reliable way in UniProt search
    # Updated fields parameter to use correct UniProt field names
    url = f"https://rest.uniprot.org/uniprotkb/search?query={clean_id}&format=json&fields=accession,protein_name,cc_function"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        
        # If the search failed with 400, it might be the fields parameter, try without it
        if response.status_code == 400:
            url_simple = f"https://rest.uniprot.org/uniprotkb/search?query={clean_id}&format=json"
            response = requests.get(url_simple, headers=headers, timeout=15)
            
        response.raise_for_status()
        data = response.json()
        
        if not data.get('results'):
            # Try searching with xref as a backup
            query_xref = f'xref:ensembl-{clean_id}'
            url_xref = f"https://rest.uniprot.org/uniprotkb/search?query={quote(query_xref)}&format=json&fields=accession,protein_name,cc_function"
            response = requests.get(url_xref, headers=headers, timeout=15)
            if response.status_code == 200:
                data = response.json()
        
        if not data.get('results'):
            return ensp_id, "Unknown", "No functional description available."
        
        # Take the first result
        result = data['results'][0]
        uniprot_id = result.get('primaryAccession', 'Unknown')
        
        # Get protein name
        protein_name = "Unknown"
        if 'proteinDescription' in result:
            desc = result['proteinDescription']
            if 'recommendedName' in desc:
                protein_name = desc['recommendedName'].get('fullName', {}).get('value', 'Unknown')
            elif 'submissionNames' in desc and len(desc['submissionNames']) > 0:
                protein_name = desc['submissionNames'][0].get('fullName', {}).get('value', 'Unknown')
            elif 'alternativeNames' in desc and len(desc['alternativeNames']) > 0:
                protein_name = desc['alternativeNames'][0].get('fullName', {}).get('value', 'Unknown')
        
        # Get functional description
        function_desc = "No functional description available."
        if 'comments' in result:
            for comment in result['comments']:
                if comment.get('commentType') == 'FUNCTION':
                    texts = comment.get('texts', [])
                    if texts:
                        function_desc = texts[0].get('value', function_desc)
                        break
        
        return ensp_id, protein_name, function_desc
        
    except Exception as e:
        return ensp_id, "Unknown", "No functional description available."

def main():
    data_dir = Path("data/processed")
    files = ["shs27k_train.csv", "shs27k_val.csv", "shs27k_test.csv"]
    
    unique_proteins = set()
    for f in files:
        path = data_dir / f
        if path.exists():
            df = pd.read_csv(path)
            unique_proteins.update(df.iloc[:, 0].astype(str).unique())
            unique_proteins.update(df.iloc[:, 1].astype(str).unique())
    
    logger.info(f"Found {len(unique_proteins)} unique proteins.")
    
    output_file = data_dir / "protein_info.csv"
    
    # Load existing info
    protein_info = {}
    if output_file.exists():
        old_df = pd.read_csv(output_file)
        for _, row in old_df.iterrows():
            # Only keep non-Unknown and non-Error results for re-fetching
            p_name = str(row['protein_name'])
            if p_name != "Unknown" and p_name != "Error":
                protein_info[str(row['protein_id'])] = (row['protein_name'], row['function'])
        logger.info(f"Loaded {len(protein_info)} valid existing protein infos. (Re-fetching Unknowns/Errors)")
    
    proteins_to_fetch = [p for p in unique_proteins if p not in protein_info]
    
    if not proteins_to_fetch:
        logger.info("All protein info already fetched.")
        return

    logger.info(f"Fetching info for {len(proteins_to_fetch)} proteins...")
    
    consecutive_unknowns = 0
    max_consecutive_unknowns = 50 # Increased threshold
    
    for i, p_id in enumerate(tqdm(proteins_to_fetch)):
        p_id, name, func = fetch_protein_info(p_id)
        protein_info[p_id] = (name, func)
        
        if name == "Unknown":
            consecutive_unknowns += 1
        else:
            consecutive_unknowns = 0
            
        if consecutive_unknowns >= max_consecutive_unknowns:
            logger.warning(f"Detected {consecutive_unknowns} consecutive Unknown results for ID starting with {p_id}. Something might be wrong with the query logic. Stopping for manual check.")
            break
            
        # Be nice to the API
        time.sleep(0.1)
        
        # Save every 50 proteins
        if (i + 1) % 50 == 0:
            final_df = pd.DataFrame([
                {'protein_id': k, 'protein_name': v[0], 'function': v[1]}
                for k, v in protein_info.items()
            ])
            final_df.to_csv(output_file, index=False)
    
    final_df = pd.DataFrame([
        {'protein_id': k, 'protein_name': v[0], 'function': v[1]}
        for k, v in protein_info.items()
    ])
    final_df.to_csv(output_file, index=False)
    logger.info(f"Saved protein info to {output_file}")

if __name__ == "__main__":
    main()
