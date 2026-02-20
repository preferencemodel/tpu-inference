import time
from pathlib import Path
from vllm import LLM, SamplingParams


def get_ctx(file_path: Path) -> str:
    with open(file_path, "r") as file:
        return file.read()


def ask(llm: LLM, context: str, question: str, params: SamplingParams) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer questions based only on "
                "the provided book text. Be concise."
            ),
        },
        {
            "role": "user",
            "content": f"Here is the full text of a book:\n\n{context}\n\nQuestion: {question}",
        },
    ]
    outputs = llm.chat(messages, params)
    return outputs[0].outputs[0].text.strip()


if __name__ == "__main__":
    book_path = Path("./metamorphosis_franz_kafka.txt")
    context = get_ctx(book_path)
    print(f"Loaded {len(context)} characters from {book_path}")

    llm = LLM(
        model="../models/gemma3-1b-it",
        tensor_parallel_size=1,
        max_model_len=32768,
        block_size=64,
        enable_prefix_caching=True,
    )
    params = SamplingParams(temperature=0.7, max_tokens=2048)

    questions = [
        # Gregor Samsa wakes up transformed into a giant insect, struggles to support
        # his family who gradually abandons him, dies alone, family feels relieved and
        # moves on with their lives
        "Provide summary of the book.",
        # He wakes up and finds himself transformed into a giant insect/vermin,
        # lying on his hard shell-like back, unable to get out of bed
        "What does Gregor Samsa discover when he wakes up at the beginning of the story?",
        # Gregor dies from an apple lodged in his back thrown by his father, starved
        # and weakened. After his death the family feels relieved, takes a tram trip
        # to the countryside and starts planning a better future
        "How does Gregor Samsa die and what happens to his family after his death?",
        # Initially most caring and devoted, brings him food and cleans his room.
        # Gradually becomes resentful, ultimately declares to the family that
        # they must get rid of him — triggering his final decline
        "How does Gregor's sister Grete change her attitude toward him throughout the story?",
        # Beginning: Gregor was the sole breadwinner supporting the whole family.
        # After transformation: father, mother and Grete all take up jobs.
        # End: family is financially stable and independent, relieved to be free
        "How does the family's financial situation change from the beginning to the end of the story?",
    ]

    start = time.time()
    for q in questions:
        print(f"\nQ: {q}")
        answer = ask(llm, context, q, params)
        print(f"A: {answer}")
    end = time.time()
    print(f"\nTime: {end - start:.2f}s")