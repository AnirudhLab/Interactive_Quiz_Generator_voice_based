import openai

def generate_quiz_from_doc(context, num_questions, difficulty, api_key):
    prompt = f"""
You are a helpful quiz generator. Based on the following context, generate {num_questions} multiple choice questions (MCQs) with 4 options each. Ensure only one correct answer per question. Include an explanation.

Context:
{context}

Respond in the following JSON format:
[
  {{
    "question": "...",
    "options": {{
      "A": "...",
      "B": "...",
      "C": "...",
      "D": "..."
    }},
    "correct": "B",
    "explanation": "..."
  }},
  ...
]
Ensure all output is valid JSON.
Difficulty: {difficulty}
"""
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content
