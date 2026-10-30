import os
import re
import csv
import json
import requests
import zipfile
import io
import time
from datetime import datetime
from dateutil import parser
import pandas as pd
import numpy as np
from tqdm import tqdm
import networkx as nx

try:
    import faiss
except ImportError:
    faiss = None

try:
   
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    from sklearn.feature_extraction.text import TfidfVectorizer

def get_base_dir():
    return os.path.dirname(os.path.abspath(__file__))

def clean_actor(actor):
    if pd.isna(actor):
        return ""
    return str(actor).strip()

def process_acled():
    print("=== STEP 2: Load ACLED Data ===")
    acled_triples = []
    filepath = os.path.join(get_base_dir(), "kg_data", "acled_data.csv")
    if not os.path.exists(filepath):
        print(f"Warning: ACLED file not found at {filepath}")
        return acled_triples

    df = pd.read_csv(filepath, low_memory=False)
    df.columns = [str(c).lower() for c in df.columns]

    target_event_types = ["Battles", "Explosions/Remote violence", "Strategic developments"]
    if "event_type" in df.columns:
        df = df[df["event_type"].isin(target_event_types)]

    for _, row in df.iterrows():
        actor = clean_actor(row.get("actor1"))
        if not actor:
            continue
            
        sub_event_type = str(row.get("sub_event_type", ""))
        relation = sub_event_type.lower().replace(" ", "_")
        
        actor2 = clean_actor(row.get("actor2"))
        event_type = str(row.get("event_type", ""))
        object_ = actor2 if actor2 else event_type
        
        loc = str(row.get("location", ""))
        country = str(row.get("country", ""))
        location = f"{loc}, {country}" if loc and country else (loc or country)
        
        event_date_str = str(row.get("event_date", ""))
        try:
            date_parsed = parser.parse(event_date_str).strftime("%Y-%m-%d")
        except:
            date_parsed = event_date_str

        acled_triples.append({
            "actor": actor,
            "relation": relation,
            "object": object_,
            "location": location,
            "date": date_parsed,
            "source": "ACLED"
        })
        
    print(f"Extracted {len(acled_triples)} ACLED triples.")
    return acled_triples

def process_sipri():
    print("=== STEP 3: Load SIPRI Data ===")
    sipri_triples = []
    filepath = os.path.join(get_base_dir(), "kg_data", "sipri_transfers.csv")
    if not os.path.exists(filepath):
        print(f"Warning: SIPRI file not found at {filepath}")
        return sipri_triples

    df = pd.read_csv(filepath, low_memory=False)
    df.columns = [str(c).lower() for c in df.columns]
    
    if "designation" in df.columns and "armament" not in df.columns:
        df["armament"] = df["designation"]
    if "order_year" in df.columns and "year" not in df.columns:
        df["year"] = df["order_year"]
        
    for _, row in df.iterrows():
        supplier = clean_actor(row.get("supplier"))
        recipient = clean_actor(row.get("recipient"))
        if not supplier or not recipient or str(supplier).lower() == "nan" or str(recipient).lower() == "nan":
            continue
            
        armament = str(row.get("armament", ""))
        year = str(row.get("year", ""))
        
        actor = supplier
        relation = "transferred_to"
        object_ = f"{armament} to {recipient}"
        location = recipient
        try:
            year_int = int(float(year))
            date_parsed = f"{year_int}-01-01"
        except:
            date_parsed = f"{year}-01-01" if year else ""
        
        sipri_triples.append({
            "actor": actor,
            "relation": relation,
            "object": object_,
            "location": location,
            "date": date_parsed,
            "source": "SIPRI"
        })
        
    print(f"Extracted {len(sipri_triples)} SIPRI triples.")
    return sipri_triples

def process_gdelt():
    print("=== STEP 4: Load GDELT Data ===")
    gdelt_triples = []
    try:
        url = "http://data.gdeltproject.org/events/LASTUPDATE.TXT"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        latest_zip_url = response.text.split()[2]
        
        print(f"Downloading latest GDELT data from {latest_zip_url}...")
        zip_resp = requests.get(latest_zip_url, timeout=30)
        zip_resp.raise_for_status()
        
        with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as z:
            csv_filename = z.namelist()[0]
            with z.open(csv_filename) as f:
                df = pd.read_csv(f, sep="\t", header=None, low_memory=False)
                
        df_filtered = df[df[27].isin([18, 19, 20, "18", "19", "20", 18.0, 19.0, 20.0])]
        
        for _, row in df_filtered.iterrows():
            actor1 = clean_actor(row.get(4))
            if not actor1 or str(actor1).lower() == "nan":
                continue
                
            sqldate = str(row.get(1))
            try:
                date_parsed = f"{sqldate[:4]}-{sqldate[4:6]}-{sqldate[6:8]}"
            except:
                date_parsed = sqldate
                
            obj = clean_actor(row.get(17))
            loc = str(row.get(51))
            if loc.lower() == "nan": loc = ""
            if obj.lower() == "nan": obj = ""

            gdelt_triples.append({
                "actor": actor1,
                "relation": "conflict_event",
                "object": obj,
                "location": loc,
                "date": date_parsed,
                "source": "GDELT"
            })
            
    except Exception as e:
        print(f"Warning: Failed to load GDELT data. Error: {e}")
        
    print(f"Extracted {len(gdelt_triples)} GDELT triples.")
    return gdelt_triples

def merge_triples(acled, sipri, gdelt):
    print("=== STEP 5: Merge All Triples ===")
    all_triples = acled + sipri + gdelt
    df = pd.DataFrame(all_triples, columns=["actor", "relation", "object", "location", "date", "source"])
    
    df = df[df["actor"].astype(bool)]
    df = df[df["actor"].str.strip() != ""]
    df = df[df["actor"].str.lower() != "nan"]
    df = df.drop_duplicates(subset=["actor", "relation", "object", "location", "date"])
    df = df.fillna("")
    
    kg_path = os.path.join(get_base_dir(), "kg_data", "miltfact_kg.csv")
    df.to_csv(kg_path, index=False)
    print(f"Total triples after deduplication: {len(df)}")
    
    G = nx.DiGraph()
    for _, row in df.iterrows():
        actor = row["actor"]
        obj = row["object"]
        rel = row["relation"]
        G.add_node(actor)
        G.add_node(obj)
        G.add_edge(actor, obj, relation=rel, location=row["location"], date=row["date"], source=row["source"])
        
    graph_path = os.path.join(get_base_dir(), "kg_data", "miltfact_kg.graphml")
    nx.write_graphml(G, graph_path)
    
    return df

class NumpyVectorIndex:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.ntotal = len(embeddings)
        self.d = embeddings.shape[1] if len(embeddings) > 0 else 0

    def search(self, query_emb, top_k):
        sims = np.dot(self.embeddings, query_emb.T).squeeze()
        if sims.ndim == 0:
            sims = np.array([sims])
        top_k = min(top_k, len(sims))
        indices = np.argsort(sims)[::-1][:top_k]
        scores = sims[indices]
        return np.array([scores]), np.array([indices])

def build_faiss_index(df):
    print("=== STEP 6: Build Knowledge-Graph Vector Index ===")
    sentences = []
    for _, row in df.iterrows():
        sentences.append(f"{row['actor']} {row['relation']} {row['object']} in {row['location']} on {row['date']}")
        
    sentences_path = os.path.join(get_base_dir(), "kg_data", "kg_sentences.txt")
    with open(sentences_path, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s + "\n")

    if SENTENCE_TRANSFORMERS_AVAILABLE:
        print("Loading SentenceTransformer model...")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("Embedding sentences with SentenceTransformer...")
        embeddings = model.encode(sentences, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    else:
        print("Using TF-IDF Fallback Vectorizer for Sentence Embeddings...")
        model = TfidfVectorizer(max_features=384)
        embeddings = model.fit_transform(sentences).toarray()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        embeddings = embeddings / norms

    dim = embeddings.shape[1]
    if faiss is not None:
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))
        index_path = os.path.join(get_base_dir(), "kg_data", "miltfact_faiss.index")
        faiss.write_index(index, index_path)
        print(f"FAISS index built with {index.ntotal} vectors of dimension {dim}")
    else:
        index = NumpyVectorIndex(embeddings)
        print(f"Numpy fallback index built with {index.ntotal} vectors of dimension {dim}")
    
    return model, index, sentences

def retrieve_candidates(query_claim, model, index, sentences, top_k=20):
    if hasattr(model, 'encode'):
        query_emb = model.encode([query_claim], normalize_embeddings=True)
    else:
        query_emb = model.transform([query_claim]).toarray()
        norm = np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-9
        query_emb = query_emb / norm

    D, I = index.search(query_emb, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        results.append((sentences[idx], score))
    return results

def test_pipeline(model, index, sentences, report_data):
    print("=== STEP 7: Test Retrieval Pipeline ===")
    queries = [
        "Russian forces attacked civilian infrastructure in Kharkiv",
        "Ukraine received Javelin missiles from the United States",
        "Israeli military conducted airstrikes in Gaza"
    ]
    
    results_str = ""
    for i, q in enumerate(queries):
        print(f"\nQuery {i+1}: {q}")
        results_str += f"\nQuery {i+1}: {q}\n"
        results = retrieve_candidates(q, model, index, sentences, top_k=5)
        for s, score in results:
            print(f"  [Score: {score:.4f}] {s}")
            results_str += f"  [Score: {score:.4f}] {s}\n"
            
    report_data["test_results"] = results_str

def write_report(report_data):
    print("=== STEP 8: Generate Summary Report ===")
    report_path = os.path.join(get_base_dir(), "phase1_output", "phase2_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Phase 2 Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"ACLED triple count: {report_data['acled_count']}\n")
        f.write(f"SIPRI triple count: {report_data['sipri_count']}\n")
        f.write(f"GDELT triple count: {report_data['gdelt_count']}\n")
        f.write(f"Total triples after deduplication: {report_data['total_count']}\n")
        f.write(f"FAISS index size: {report_data['faiss_size']}\n")
        f.write(f"FAISS embedding dimension: {report_data['faiss_dim']}\n")
        f.write("\nTest Retrieval Results:\n")
        f.write(report_data['test_results'])

def main():
    report_data = {"warnings": ""}
    
    acled = process_acled()
    report_data["acled_count"] = len(acled)
    
    sipri = process_sipri()
    report_data["sipri_count"] = len(sipri)
    
    gdelt = process_gdelt()
    report_data["gdelt_count"] = len(gdelt)
    
    df = merge_triples(acled, sipri, gdelt)
    report_data["total_count"] = len(df)
    
    model, index, sentences = build_faiss_index(df)
    report_data["faiss_size"] = getattr(index, 'ntotal', len(sentences))
    report_data["faiss_dim"] = getattr(index, 'd', 384)
    
    test_pipeline(model, index, sentences, report_data)
    
    write_report(report_data)
    print("\nPhase 2 Complete.")

if __name__ == "__main__":
    main()
