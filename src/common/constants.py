# constants.py
PDF_PROMPT_TEMPLATE = """
You are a helpful assistant. Use ONLY the following context to answer the question.
If the answer is not in the context, respond with "I don't know."

Context:
{context}

Question: {question}
Answer:
"""