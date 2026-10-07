import os, time
import numpy as np
from openai import OpenAI
from rag_pipeline import RAGPipeline

EVAL_QUERIES = [
 {"question":"How do I fix login issues?","relevant_ticket_ids":["TICK-001","TICK-006"]},
 {"question":"What database problems have we seen?","relevant_ticket_ids":["TICK-002","TICK-007"]},
 {"question":"Why are emails delayed?","relevant_ticket_ids":["TICK-003","TICK-010"]},
 {"question":"What happened with payments?","relevant_ticket_ids":["TICK-005"]},
]

def calculate_metrics(retrieved_ids,relevant_ids,k=3):
    retrieved_set=set(retrieved_ids[:k]); relevant_set=set(relevant_ids)
    tp=len(retrieved_set & relevant_set)
    precision=tp/k if k else 0
    recall=tp/len(relevant_set) if relevant_set else 0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0
    return {"precision":precision,"recall":recall,"f1":f1}

def average_precision(retrieved_ids,relevant_ids):
    relevant_set=set(relevant_ids); hits=0; values=[]
    for k,doc_id in enumerate(retrieved_ids,1):
        if doc_id in relevant_set:
            hits += 1; values.append(hits/k)
    return float(np.mean(values)) if values else 0.0

def evaluate_groundedness(answer,docs):
    client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    context="\n\n".join(d.page_content for d in docs)
    prompt=f"""Evaluate whether the ANSWER is supported by the CONTEXT.
Rate groundedness from 0 to 10 and explain briefly.
CONTEXT:
{context}
ANSWER:
{answer}
Format: Score: X / Reason: <explanation>"""
    r=client.chat.completions.create(model=os.getenv("OPENAI_CHAT_MODEL","gpt-4o-mini"),
        messages=[{"role":"user","content":prompt}],temperature=0)
    return r.choices[0].message.content

if __name__ == "__main__":
    rag=RAGPipeline()
    for k in [1,3,5]:
        scores=[]
        for item in EVAL_QUERIES:
            docs=rag.retrieve(item["question"],k)
            ids=[d.metadata["ticket_id"] for d in docs]
            m=calculate_metrics(ids,item["relevant_ticket_ids"],k)
            m["ap"]=average_precision(ids,item["relevant_ticket_ids"]); scores.append(m)
        print(f"k={k} Precision={np.mean([x['precision'] for x in scores]):.3f} "
              f"Recall={np.mean([x['recall'] for x in scores]):.3f} "
              f"F1={np.mean([x['f1'] for x in scores]):.3f} "
              f"AP={np.mean([x['ap'] for x in scores]):.3f}")
