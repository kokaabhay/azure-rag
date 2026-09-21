from app.llm.query_rewriter import rewrite_query
from app.llm.prompt import build_prompt
from app.llm.service import generate_answer
from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.reranker import rerank_documents


def main():
    query = input("Ask a question: ")

    rewritten_query = rewrite_query(query)

    print("\nOriginal query:")
    print(query)

    print("\nRewritten query:")
    print(rewritten_query)

    documents = hybrid_search(rewritten_query)

    print(f"\nRetrieved {len(documents)} documents.")

    reranked_documents = rerank_documents(
        query=query,
        documents=documents,
        top_k=5,
    )

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
        documents=reranked_documents,
    )

    answer = generate_answer(prompt)

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(answer)


if __name__ == "__main__":
    main()