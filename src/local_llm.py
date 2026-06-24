import requests


class LocalLLM:
    """
    Local LLM wrapper using Ollama.

    This class talks to the Ollama server running locally at:
        http://localhost:11434

    It is used for two different tasks:
    1. Query rewriting before retrieval.
    2. Final answer generation after retrieval.
    """

    def __init__(self, model_name="gemma3:1b"):
        self.model_name = model_name
        self.url = "http://localhost:11434/api/generate"

        print(f"Using local Ollama model: {self.model_name}")
        print(f"Ollama endpoint: {self.url}")

    def generate(self, prompt, max_new_tokens=250, temperature=0.2):
        """
        Send a prompt to the local Ollama model and return generated text.
        """

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": temperature
            }
        }

        response = requests.post(
            self.url,
            json=payload,
            timeout=180
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama request failed with status {response.status_code}\n"
                f"{response.text}"
            )

        data = response.json()

        return data["response"].strip()

    def rewrite_query(self, conversation_context, user_query):
        """
        Rewrite a possibly vague follow-up question into a standalone question.

        Example:
            Conversation:
                User: What are convolution kernels?
                Assistant: Convolution kernels are small matrices used in CNNs.

            Latest question:
                How is this used in neural networks?

            Output:
                How are convolution kernels used in neural networks?
        """

        prompt = f"""
You are a query rewriting module for a retrieval-augmented generation system.

Rewrite the latest user question into a standalone question for retrieval.

Your job is NOT to answer the question.
Your job is NOT to shorten the question into keywords.

Rules:
1. Preserve the user's original intent.
2. Preserve the question type:
   - If the user asks "what", keep it a "what" question.
   - If the user asks "how", keep it a "how" question.
   - If the user asks for examples, keep it asking for examples.
   - If the user asks for comparison, keep it asking for comparison.
   - If the user asks for types, keep it asking for types.
3. Resolve pronouns such as "this", "that", "it", "they", and "them" using the conversation.
4. Replace vague references with the actual technical concept from the conversation.
5. Keep the rewritten query as a complete natural-language question.
6. Do not output bullet points.
7. Do not explain your rewrite.
8. Output only the standalone question.

Examples:

Conversation:
User: What are convolution kernels?
Assistant: Convolution kernels are small matrices used in CNNs.
Latest question:
How is this used in neural networks?
Standalone question:
How are convolution kernels used in neural networks?

Conversation:
User: What is attention?
Assistant: Attention maps queries, keys, and values to an output.
Latest question:
How is this used in transformers?
Standalone question:
How is attention used in transformers?

Conversation:
User: What is random forest?
Assistant: A random forest is an ensemble of decision trees.
Latest question:
What are its advantages?
Standalone question:
What are the advantages of random forests?

Conversation:
User: What is scaled dot-product attention?
Assistant: Scaled dot-product attention computes attention weights using query-key dot products.
Latest question:
Give examples of this.
Standalone question:
Give examples of scaled dot-product attention.

Now rewrite the actual latest question.

Conversation:
{conversation_context}

Latest question:
{user_query}

Standalone question:
""".strip()

        rewritten = self.generate(
            prompt=prompt,
            max_new_tokens=100,
            temperature=0.0
        )

        rewritten = rewritten.strip()

        # Clean accidental labels from the model output.
        prefixes = [
            "Standalone question:",
            "Standalone search query:",
            "Rewritten query:",
            "Search query:",
            "Query:",
            "Answer:"
        ]

        for prefix in prefixes:
            if rewritten.lower().startswith(prefix.lower()):
                rewritten = rewritten[len(prefix):].strip()

        # Keep only first line if model outputs extra explanation.
        rewritten = rewritten.split("\n")[0].strip()

        # Safety fallback: if the model returns nothing, use original query.
        if not rewritten:
            return user_query

        return rewritten