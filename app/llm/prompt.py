def build_prompt(query: str, documents: list[dict]) -> str:
    context_parts = []

    for i, document in enumerate(documents, start=1):
        context_parts.append(
            f"[Source {i}: {document['source']}]\n"
            f"{document['content']}"
        )

    context = "\n\n".join(context_parts)

    return f"""
You are a helpful customer support assistant.

Answer the user's question using ONLY the information
provided in the knowledge base context below.

If the answer cannot be found in the knowledge base,
say that you do not have enough information to answer
the question.

Be concise, polite, and easy to understand.

Knowledge Base Context:
------------------------
{context}
------------------------

User Question:
{query}

Instructions:
- Do not invent information.
- Do not use outside knowledge.
- Answer directly.
- When possible, mention the source document.
""".strip()