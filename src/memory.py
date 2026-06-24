VAGUE_WORDS = {
    "this",
    "that",
    "it",
    "they",
    "them",
    "these",
    "those",
    "above",
    "same"
}


class ConversationState:
    def __init__(self):
        self.history = []

    def add_user(self, query):
        self.history.append({
            "role": "user",
            "content": query
        })

    def add_assistant(self, answer):
        self.history.append({
            "role": "assistant",
            "content": answer
        })

    def recent_context(self, n=4):
        recent_messages = self.history[-n:]

        formatted = []

        for message in recent_messages:
            formatted.append(f"{message['role']}: {message['content']}")

        return "\n".join(formatted)


def is_followup(query):
    query_words = set(query.lower().replace("?", "").split())

    return len(query_words.intersection(VAGUE_WORDS)) > 0


def rewrite_query(query, state: ConversationState):
    if not is_followup(query):
        return query

    recent_context = state.recent_context(n=4)

    rewritten = f"""
Conversation context:
{recent_context}

Follow-up question:
{query}

Rewrite this follow-up as a standalone retrieval query.
""".strip()

    return rewritten