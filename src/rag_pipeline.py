import re

from src.vector_store import load_vector_index, vector_search
from src.knowledge_graph import load_kg, graph_retrieve
from src.memory import ConversationState
from src.local_llm import LocalLLM


def clean_text(text):
    """
    Normalize whitespace in retrieved document text.
    """

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_compact_prompt(user_query, rewritten_query, doc_chunks, kg_triples, state):
    """
    Build the final prompt given to the answer-generation LLM.

    This prompt contains:
    1. Conversation context.
    2. Original user question.
    3. Rewritten standalone question.
    4. Retrieved document chunks.
    5. Retrieved KG triples.
    """

    document_parts = []

    for i, chunk in enumerate(doc_chunks[:3], start=1):
        meta = chunk["metadata"]
        text = clean_text(chunk["text"])[:900]
        score = chunk.get("score", None)

        if score is not None:
            source_header = (
                f"Document {i}: {meta['filename']}, "
                f"page {meta['page']}, "
                f"similarity score {score:.4f}"
            )
        else:
            source_header = (
                f"Document {i}: {meta['filename']}, "
                f"page {meta['page']}"
            )

        document_parts.append(
            f"{source_header}\n{text}"
        )

    if document_parts:
        document_evidence = "\n\n".join(document_parts)
    else:
        document_evidence = "No document evidence found."

    if kg_triples:
        graph_evidence = "\n".join(
            f"{subject} --{relation}--> {obj}"
            for subject, relation, obj in kg_triples[:12]
        )
    else:
        graph_evidence = "No graph evidence found."

    prompt = f"""
You are a technical assistant answering from retrieved evidence.

Rules:
1. Answer the original user question directly.
2. Use the retrieved document evidence and knowledge graph evidence.
3. Do not answer only by defining the main topic unless the user asked for a definition.
4. If the user asks how something is used, explain its role or mechanism.
5. If the user asks for examples, provide examples.
6. If the user asks for types, list the types explicitly.
7. Do not repeat the same generic sentence from the previous answer.
8. If the evidence is insufficient, say what is missing.
9. Keep the answer clear and technical.

Conversation:
{state.recent_context(n=4)}

Original user question:
{user_query}

Standalone rewritten question:
{rewritten_query}

Retrieved document evidence:
{document_evidence}

Knowledge graph evidence:
{graph_evidence}

Answer:
""".strip()

    return prompt


class OnPremKGRAG:
    """
    Main local KG-RAG system.

    Components:
    - FAISS vector index for dense document retrieval.
    - NetworkX knowledge graph for symbolic retrieval.
    - ConversationState for multi-turn memory.
    - Ollama LocalLLM for query rewriting and answer generation.
    """

    def __init__(self):
        print("Loading vector index...")
        self.documents, self.index, self.embedding_model = load_vector_index()

        print("Loading knowledge graph...")
        self.kg = load_kg()

        print("Initializing conversation memory...")
        self.state = ConversationState()

        print("Initializing local LLM...")
        self.llm = LocalLLM()

    def ask(self, query, k=5):
        """
        Answer a user query.

        Flow:
        1. Save user query into memory.
        2. Use Ollama to rewrite query into standalone question.
        3. Retrieve document chunks using FAISS.
        4. Retrieve symbolic triples using KG.
        5. Build final evidence prompt.
        6. Use Ollama to generate answer.
        7. Save answer into memory.
        """

        self.state.add_user(query)

        conversation_context = self.state.recent_context(n=4)

        rewritten_query = self.llm.rewrite_query(
            conversation_context=conversation_context,
            user_query=query
        )

        print(f"\n[Rewritten query]: {rewritten_query}\n")

        doc_chunks = vector_search(
            query=rewritten_query,
            documents=self.documents,
            index=self.index,
            embedding_model=self.embedding_model,
            k=k
        )

        print("[Retrieved document chunks]:")
        for i, chunk in enumerate(doc_chunks, start=1):
            meta = chunk["metadata"]
            score = chunk.get("score", None)

            if score is not None:
                print(
                    f"{i}. {meta['filename']} | "
                    f"page {meta['page']} | "
                    f"score {score:.4f}"
                )
            else:
                print(
                    f"{i}. {meta['filename']} | "
                    f"page {meta['page']}"
                )
        print()

        kg_triples = graph_retrieve(
            query=rewritten_query,
            kg=self.kg
        )

        print("[Retrieved KG triples]:")
        if kg_triples:
            for subject, relation, obj in kg_triples[:12]:
                print(f"- {subject} --{relation}--> {obj}")
        else:
            print("- No graph evidence found.")
        print()

        prompt = build_compact_prompt(
            user_query=query,
            rewritten_query=rewritten_query,
            doc_chunks=doc_chunks,
            kg_triples=kg_triples,
            state=self.state
        )

        answer = self.llm.generate(
            prompt=prompt,
            max_new_tokens=350,
            temperature=0.2
        )

        self.state.add_assistant(answer)

        return answer

    def reset(self):
        """
        Reset multi-turn conversation memory.
        """

        self.state = ConversationState()