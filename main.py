
import time
import ollama

MODEL = "qwen2.5:3b"
TOKEN_LIMIT = 1000

SYSTEM_MESSAGE = (
    "Tu es un assistant pédagogique. "
    "Tu expliques simplement et tu gardes le contexte de la conversation. "
    "Utilise les questions et réponses précédentes pour comprendre "
    "les questions suivantes."
)

# Historique de la conversation
messages = [
    {"role": "system", "content": SYSTEM_MESSAGE}
]

# Compteurs de consommation
total_input_tokens = 0
total_output_tokens = 0
total_duration = 0.0

print("=== Assistant IA avec mémoire ===")
print("Tape 'quit' pour quitter.\n")

while True:
    # Bloquer les questions si la limite est atteinte
    if total_input_tokens + total_output_tokens >= TOKEN_LIMIT:
        print("⚠️ Limite de tokens atteinte")
        break

    question = input("Toi : ").strip()

    if question.lower() == "quit":
        break

    if not question:
        continue

    messages.append({
        "role": "user",
        "content": question
    })

    start_time = time.perf_counter()

    try:
        response = ollama.chat(
            model=MODEL,
            messages=messages,
            options={
                "temperature": 0.2,
                "num_predict": 1000
            }
        )

        duration = time.perf_counter() - start_time
        answer = response["message"]["content"]

        # Récupérer les statistiques de tokens
        input_tokens = response.get("prompt_eval_count", 0)
        output_tokens = response.get("eval_count", 0)

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        total_duration += duration

        # Conserver la réponse dans l'historique
        messages.append({
            "role": "assistant",
            "content": answer
        })

        print(f"\nIA : {answer}\n")

        # Vérifier la limite après la réponse
        if total_input_tokens + total_output_tokens >= TOKEN_LIMIT:
            print("⚠️ Limite de tokens atteinte")
            break

    except Exception as error:
        messages.pop()
        print(f"\nErreur : {error}\n")

print("\n=== Bilan final ===")
print(f"Total tokens : {total_input_tokens + total_output_tokens}")
print(f"Temps cumulé : {total_duration:.2f} secondes")
print("Fin du programme.")
