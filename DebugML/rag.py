
import ollama

def generate_rag_answer(query, results):
    context = ""
    for r in results:
        meta = r["metadata"]
        context += f"Failure: {meta['text']}\nFix: {meta['fix']}\n\n"

    prompt = f'''
You are an expert ML engineer.
User issue: {query}

Past similar failures and fixes:
{context}

Explain the likely cause and suggest solutions.
'''
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]
