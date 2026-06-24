from src.rag_pipeline import OnPremKGRAG


def main():
    rag = OnPremKGRAG()

    print("\nLocal On-Prem KG-RAG system is ready.")
    print("Type 'exit' or 'quit' to quit.")
    print("Type 'reset' to clear conversation memory.\n")

    while True:
        query = input("You: ").strip()

        if not query:
            continue

        if query.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        if query.lower() == "reset":
            rag.reset()
            print("Conversation memory reset.\n")
            continue

        answer = rag.ask(query, k=5)

        print("\nAssistant:")
        print(answer)
        print()


if __name__ == "__main__":
    main()