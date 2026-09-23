from fastapi import FastAPI,HTTPException,APIRouter
from api.response_object import Response_Object
from pydantic import Field,BaseModel
from app.llm.query_rewriter import rewrite_query
from app.llm.prompt import build_prompt
from app.llm.service import generate_answer
from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.reranker import rerank_documents
from app.llm.prompt import build_context
router=APIRouter(tags=["API"])




@router.post("/get_response")
def response(response_object:Response_Object):
    query=response_object.query
    if not query:
        raise HTTPException
    rewritten_query = rewrite_query(query)
    
    print("\nOriginal query:")
    print(query)

    print("\nRewritten query:")
    print(rewritten_query)

    documents = hybrid_search(rewritten_query)

    print(f"\nRetrieved {len(documents)} documents.")
    # for i in documents:
    #     print(i)
    reranked_documents = rerank_documents(
        query=query,
        documents=documents,
        top_k=5,
    )
    
    context_documents=build_context(reranked_documents)
    #print("context documents: ",context_documents)
    print("\nReranked documents:\n")

    for i, document in enumerate(
        reranked_documents,
        start=1,
    ):
        print(f"--- Result {i} ---")
        print(f"Source: {document['source']}")
        print(f"Search score: {document['score']}")
        print(f"Rerank score: {document['rerank_score']}")
        print(document["content"][:500])
        print()

    prompt = build_prompt(
        query=query,
        documents=context_documents,
    )

    answer = generate_answer(prompt)

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(answer)
    return answer
    