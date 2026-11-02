"""Build a candidate pool without treating candidates as relevance labels."""
import json,time
from pathlib import Path
from ml.retrieval.engine import Retriever


def generate(queries, output, k=5):
    output=Path(output)
    if output.exists(): raise FileExistsError(f'Output already exists: {output}')
    start=time.perf_counter(); engine=Retriever(); load=time.perf_counter()-start
    pool={}; latency={}
    for query_id,text in queries.items():
        pool[query_id]={}; latency[query_id]={}
        for method in ('tfidf','word2vec'):
            start=time.perf_counter(); pool[query_id][method]=engine.search(text,method,k)
            latency[query_id][method]=time.perf_counter()-start
    output.mkdir(parents=True)
    (output/'queries.json').write_text(json.dumps(queries,indent=2),encoding='utf-8')
    (output/'candidate_pool.json').write_text(json.dumps(pool,indent=2,ensure_ascii=False),encoding='utf-8')
    (output/'execution_metadata.json').write_text(json.dumps({'documents':len(engine.records),'collection_order':['train','validation','test'],'initial_load_seconds':load,'query_seconds':latency,'tfidf_shape':engine.tfidf.shape,'word2vec_shape':engine.word2vec.shape},indent=2),encoding='utf-8')
    return pool
