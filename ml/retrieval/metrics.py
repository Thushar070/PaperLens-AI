"""Metrics over explicit judgments; unjudged results count as nonrelevant."""
import math


def evaluate(ranked, judgments, k):
    if not isinstance(k,int) or k<1: raise ValueError('k must be positive')
    if len(set(ranked))!=len(ranked): raise ValueError('Duplicate ranked IDs')
    if not judgments or any(type(v) is not int or v not in (0,1,2) for v in judgments.values()): raise ValueError('Judgments must contain grades 0, 1 or 2')
    relevant={key for key,value in judgments.items() if value>0}
    if not relevant: raise ValueError('Query has no judged relevant documents')
    top=ranked[:k]; hits=sum(item in relevant for item in top)
    rr=next((1/(i+1) for i,item in enumerate(ranked) if item in relevant),0.0)
    dcg=sum((2**judgments.get(item,0)-1)/math.log2(i+2) for i,item in enumerate(top))
    ideal=sorted(judgments.values(),reverse=True)[:k]
    idcg=sum((2**grade-1)/math.log2(i+2) for i,grade in enumerate(ideal))
    return {'precision':hits/k,'recall':hits/len(relevant),'mrr':rr,'ndcg':dcg/idcg,'unjudged_in_top_k':sum(item not in judgments for item in top)}
