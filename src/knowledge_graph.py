import pickle
import networkx as nx

from src.config import KG_PATH


ML_TRIPLES = [
    # Attention basics
    ("attention", "maps", "queries keys and values to an output"),
    ("attention", "computes", "weighted combinations of values"),
    ("attention", "uses", "queries"),
    ("attention", "uses", "keys"),
    ("attention", "uses", "values"),

    # Transformer attention
    ("transformer", "uses", "self-attention"),
    ("transformer", "uses", "multi-head attention"),
    ("transformer", "uses", "scaled dot-product attention"),
    ("self-attention", "is used in", "transformers"),
    ("multi-head attention", "is used in", "transformers"),
    ("scaled dot-product attention", "is used in", "transformers"),

    # Attention types
    ("attention", "has type", "self-attention"),
    ("attention", "has type", "cross-attention"),
    ("attention", "has type", "causal attention"),
    ("attention", "has type", "multi-head attention"),
    ("attention", "has type", "scaled dot-product attention"),
    ("attention", "has type", "additive attention"),
    ("attention", "has type", "dot-product attention"),

    # Attention type explanations
    ("self-attention", "relates", "tokens within the same sequence"),
    ("cross-attention", "relates", "queries from one sequence to keys and values from another sequence"),
    ("causal attention", "prevents", "tokens from attending to future tokens"),
    ("multi-head attention", "runs", "several attention heads in parallel"),
    ("scaled dot-product attention", "computes", "softmax of query-key dot products"),
    ("additive attention", "uses", "a learned feed-forward scoring function"),
    ("dot-product attention", "uses", "query-key dot products as scores"),

    # Convolution and CNNs
    ("convolution kernel", "is used in", "convolutional neural network"),
    ("convolution kernel", "extracts", "local features"),
    ("convolution kernel", "slides over", "input feature maps"),
    ("convolution kernel", "computes", "dot products with local patches"),
    ("convolutional layer", "uses", "convolution kernels"),
    ("convolutional neural network", "uses", "convolutional layers"),
    ("convolutional neural network", "learns", "hierarchical visual features"),
    ("early convolution layers", "detect", "edges and textures"),
    ("deeper convolution layers", "detect", "object parts and semantic patterns"),
    ("feature map", "is produced by", "convolution kernels"),

    # RAG
    ("RAG", "retrieves", "external documents"),
    ("RAG", "augments", "language generation"),
    ("RAG", "uses", "retrieval before generation"),
    ("FAISS", "performs", "vector similarity search"),
    ("knowledge graph", "stores", "entities and relations"),
    ("KG-RAG", "combines", "document retrieval and graph retrieval"),

    # UWB / GNN spatial intelligence
    ("UWB", "measures", "range"),
    ("UWB", "can estimate", "relative position"),
    ("UWB", "can use", "angle of arrival"),
    ("UWB", "can use", "time difference of arrival"),
    ("graph attention network", "updates", "node embeddings"),
    ("edge features", "represent", "range angle RSSI and uncertainty"),
    ("multi-agent spatial intelligence", "uses", "dynamic graphs"),
    ("dynamic graph", "represents", "agents anchors and spatial relations"),
]


def build_kg():
    """
    Build a small manually curated knowledge graph and save it to storage/kg.pkl.
    """

    kg = nx.MultiDiGraph()

    for subject, relation, obj in ML_TRIPLES:
        kg.add_node(subject)
        kg.add_node(obj)
        kg.add_edge(subject, obj, relation=relation)

    with open(KG_PATH, "wb") as f:
        pickle.dump(kg, f)

    print(
        f"Saved KG with {kg.number_of_nodes()} nodes "
        f"and {kg.number_of_edges()} edges."
    )

    return kg


def load_kg():
    """
    Load the saved knowledge graph from storage/kg.pkl.
    """

    with open(KG_PATH, "rb") as f:
        kg = pickle.load(f)

    return kg


def extract_query_entities(query, kg):
    """
    Extract KG entities that appear in the query.

    This is a simple lexical matcher.
    Example:
        query = "How is attention used in transformers?"
        extracted entities may be:
            ["attention", "transformer"]
    """

    query_lower = query.lower()

    entities = []

    for node in kg.nodes:
        node_lower = node.lower()

        if node_lower in query_lower:
            entities.append(node)

    # Remove duplicates while preserving order.
    seen = set()
    unique_entities = []

    for entity in entities:
        if entity not in seen:
            seen.add(entity)
            unique_entities.append(entity)

    return unique_entities


def graph_retrieve(query, kg, max_triples=12):
    """
    Retrieve graph triples related to entities found in the query.

    For each matched entity, return:
    1. outgoing triples: entity --relation--> object
    2. incoming triples: subject --relation--> entity
    """

    entities = extract_query_entities(query, kg)

    triples = []

    for entity in entities:
        for _, obj, data in kg.out_edges(entity, data=True):
            triples.append((entity, data["relation"], obj))

        for subject, _, data in kg.in_edges(entity, data=True):
            triples.append((subject, data["relation"], entity))

    # Remove duplicate triples.
    seen = set()
    unique_triples = []

    for triple in triples:
        if triple not in seen:
            seen.add(triple)
            unique_triples.append(triple)

    return unique_triples[:max_triples]


if __name__ == "__main__":
    build_kg()