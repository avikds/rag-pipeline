"""
RAG Pipeline

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_text_file
def load_text_file(path):
    # Read the entire file as UTF-8 while preserving
    # newlines, whitespace, and Unicode characters exactly.
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# Step 2 - load_text_directory
def load_text_directory(directory):
    # Find all .txt files, sort by filename for deterministic ordering,
    # and reuse load_text_file() for reading their contents.
    filenames = sorted(
        filename
        for filename in os.listdir(directory)
        if filename.endswith(".txt")
    )

    return [
        load_text_file(os.path.join(directory, filename))
        for filename in filenames
    ]

# Step 3 - extract_text_from_html
def extract_text_from_html(html):
    from html.parser import HTMLParser

    class VisibleTextParser(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.parts = []
            self.skip_depth = 0

        def handle_starttag(self, tag, attrs):
            if tag.lower() in {"script", "style"}:
                self.skip_depth += 1

        def handle_endtag(self, tag):
            if tag.lower() in {"script", "style"} and self.skip_depth > 0:
                self.skip_depth -= 1

        def handle_data(self, data):
            if self.skip_depth == 0:
                self.parts.append(data)

    parser = VisibleTextParser()
    parser.feed(html)
    parser.close()

    return "".join(parser.parts)

# Step 4 - normalize_text
def normalize_text(text):
    import unicodedata
    import re

    # Apply Unicode NFKC normalization, then collapse all
    # whitespace runs into a single space and trim the result.
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()

# Step 5 - make_document
def make_document(text, source, title):
    return {
        "text": text,
        "source": source,
        "title": title,
    }

# Step 6 - chunk_fixed_size
def chunk_fixed_size(text, chunk_size):
    return [
        text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size)
    ]

# Step 7 - chunk_by_tokens
def chunk_by_tokens(text, tokenizer, max_tokens):
    if not text:
        return []

    token_ids = tokenizer.encode(text)

    chunks = []
    for i in range(0, len(token_ids), max_tokens):
        chunk_ids = token_ids[i:i + max_tokens]
        chunks.append(tokenizer.decode(chunk_ids))

    return chunks

# Step 8 - chunk_by_sentences
def chunk_by_sentences(text, max_chars):
    import re

    if not text or not text.strip():
        return []

    # Split into sentences while keeping the terminating punctuation.
    sentences = re.findall(r"[^.!?]+[.!?]+|[^.!?]+$", text)

    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    chunks = []
    current = []

    for sentence in sentences:
        if not current:
            current = [sentence]
            continue

        candidate = " ".join(current + [sentence])

        if len(candidate) <= max_chars:
            current.append(sentence)
        else:
            chunks.append(" ".join(current))
            current = [sentence]

    if current:
        chunks.append(" ".join(current))

    return chunks

# Step 9 - chunk_with_overlap
def chunk_with_overlap(text, chunk_size, overlap):
    step = chunk_size - overlap

    chunks = []
    for start in range(0, len(text), step):
        chunks.append(text[start:start + chunk_size])

    return chunks

# Step 10 - attach_chunk_metadata
def attach_chunk_metadata(chunks, source):
    return [
        {
            "text": chunk,
            "source": source,
            "position": position,
            "chunk_id": f"{source}::{position}",
        }
        for position, chunk in enumerate(chunks)
    ]

# Step 11 - load_embedding_model
def load_embedding_model(model_name):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)

# Step 12 - embed_text
def embed_text(model, text):
    import numpy as np

    embedding = model.encode(text)
    return np.asarray(embedding, dtype=np.float32).reshape(-1)

# Step 13 - embed_chunks
def embed_chunks(model, chunks, batch_size=32):
    """Batch-embed a list of chunk strings or chunk dicts into a 2D float32 matrix."""
    import numpy as np

    texts = [
        chunk["text"] if isinstance(chunk, dict) else chunk
        for chunk in chunks
    ]

    # Return an empty 2D matrix with the correct embedding dimension.
    if not texts:
        dimension = model.get_sentence_embedding_dimension()
        return np.empty((0, dimension), dtype=np.float32)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
    )

    return np.asarray(embeddings, dtype=np.float32)

# Step 14 - l2_normalize
def l2_normalize(matrix):
    # Compute the L2 norm of each row.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)

    # Avoid division by zero for all-zero rows.
    safe_norms = np.where(norms == 0, 1.0, norms)

    # Return a new normalized array.
    return matrix / safe_norms

# Step 15 - save_corpus
def save_corpus(embeddings, chunks, directory):
    import os
    import json
    import numpy as np

    os.makedirs(directory, exist_ok=True)

    embeddings_path = os.path.join(directory, "embeddings.npy")
    chunks_path = os.path.join(directory, "chunks.json")

    np.save(embeddings_path, embeddings)

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    return {
        "embeddings": np.load(embeddings_path),
        "chunks": json.load(open(chunks_path, "r", encoding="utf-8")),
    }

# Step 16 - cosine_similarity_search
def cosine_similarity_search(query_vector, chunk_matrix):
    """Cosine similarity between query_vector (d,) and each row of chunk_matrix (n,d)."""

    query_norm = np.linalg.norm(query_vector)
    chunk_norms = np.linalg.norm(chunk_matrix, axis=1)

    # Avoid division by zero for zero vectors.
    safe_query_norm = query_norm if query_norm != 0 else 1.0
    safe_chunk_norms = np.where(chunk_norms == 0, 1.0, chunk_norms)

    dot_products = chunk_matrix @ query_vector

    return dot_products / (safe_query_norm * safe_chunk_norms)

# Step 17 - top_k_indices
def top_k_indices(scores, k):
    """Return indices of the k highest scores in descending order."""

    k = min(k, len(scores))

    # Stable descending sort preserves lower indices when scores are tied.
    return np.argsort(-scores, kind="stable")[:k]

# Step 18 - top_k_chunks
def top_k_chunks(scores, chunks, k):
    # Reuse top_k_indices() to select the highest-scoring positions.
    indices = top_k_indices(scores, k)

    return [
        (chunks[i], float(scores[i]))
        for i in indices
    ]

# Step 19 - retrieve
def retrieve(query, model, chunk_matrix, chunks, k):
    # Embed the query into a 1D vector.
    query_vector = embed_text(model, query)

    # Compute cosine similarity against every chunk.
    scores = cosine_similarity_search(query_vector, chunk_matrix)

    # Return the top-k chunks with their similarity scores.
    return top_k_chunks(scores, chunks, k)

# Step 20 - build_faiss_index
def build_faiss_index(chunk_matrix):
    import faiss

    dimension = chunk_matrix.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(chunk_matrix)

    return index

# Step 21 - faiss_search
def faiss_search(index, query_vector, k):
    """Return top-k (scores, indices) as 1D arrays for a single query vector."""

    query_vector = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)

    # Search all vectors so ties can be resolved deterministically.
    scores, indices = index.search(query_vector, index.ntotal)

    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)

    # Sort by descending score, then ascending index for ties.
    order = np.lexsort((indices, -scores))

    order = order[:min(k, len(order))]

    return scores[order], indices[order]

# Step 22 - compare_faiss_to_numpy
def compare_faiss_to_numpy(query_vector, chunk_matrix, index, k):
    # NumPy cosine-similarity retrieval.
    numpy_scores = cosine_similarity_search(query_vector, chunk_matrix)
    numpy_indices = top_k_indices(numpy_scores, k)

    # FAISS-style inner-product retrieval.
    import numpy as np

    q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
    _, faiss_indices = index.search(q, k)
    faiss_indices = np.asarray(faiss_indices, dtype=np.int64).reshape(-1)

    # Compare the selected indices as sets.
    return set(numpy_indices.tolist()) == set(faiss_indices.tolist())

# Step 23 - save_faiss_index
def save_faiss_index(index, path):
    """Write `index` to `path` and return the index loaded back from disk."""
    import faiss

    faiss.write_index(index, path)
    return faiss.read_index(path)

# Step 24 - build_prompt_template
def build_prompt_template():
    return (
        "Context:\n"
        "{context}\n\n"
        "Question:\n"
        "{question}\n\n"
        "Answer the question using only the provided context."
    )

# Step 25 - format_context
def format_context(retrieved):
    if not retrieved:
        return ""

    return "\n".join(
        f"[{i}] {chunk['text']} (source={chunk['source']})"
        for i, (chunk, score) in enumerate(retrieved, start=1)
    )

# Step 26 - truncate_context
def truncate_context(context, max_chars):
    if len(context) <= max_chars:
        return context

    prefix = context[:max_chars]

    # If the cut falls in the middle of a word, cut back to the
    # previous whitespace boundary.
    if max_chars < len(context) and not context[max_chars].isspace():
        boundary = prefix.rfind(" ")
        if boundary >= 0:
            return prefix[:boundary]

    return prefix

# Step 27 - add_system_instruction
def add_system_instruction(prompt):
    """Prepend a fixed system instruction to the prompt."""

    system_instruction = (
        "You are a helpful assistant. "
        "Answer the question using ONLY the provided context. "
        "If the answer is not in the context, say 'I do not know'."
    )

    return system_instruction + "\n\n" + prompt

# Step 28 - load_generator
def load_generator(model_name="sshleifer/tiny-gpt2"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

# Step 29 - generate_answer
def generate_answer(model, tokenizer, prompt, max_new_tokens=32):
    import torch

    # Seed PyTorch for deterministic generation.
    torch.manual_seed(0)

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the newly generated tokens, excluding the prompt.
    prompt_length = inputs["input_ids"].shape[1]
    new_token_ids = output_ids[0, prompt_length:]

    return tokenizer.decode(new_token_ids, skip_special_tokens=True)

# Step 30 - rag_answer
def rag_answer(query, chunks, embeddings, embed_model, generator, tokenizer, k=3):
    # Embed the query.
    query_vector = embed_text(embed_model, query)

    # Compute cosine similarity and retrieve the top-k chunks.
    scores = cosine_similarity_search(query_vector, embeddings)
    retrieved = top_k_chunks(scores, chunks, k)

    # Format the retrieved chunks into the RAG prompt.
    context = format_context(retrieved)
    prompt_template = build_prompt_template()
    prompt = prompt_template.format(
        context=context,
        question=query,
    )

    # Add the grounded system instruction.
    prompt = add_system_instruction(prompt)

    # Generate the answer.
    answer = generate_answer(
        generator,
        tokenizer,
        prompt,
    )

    # Return the answer together with the ranked source chunks.
    return {
        "answer": answer,
        "sources": [chunk for chunk, score in retrieved],
        "query": query,
    }

# Step 31 - track_source_chunk_ids
def track_source_chunk_ids(source_chunks):
    return [
        chunk["id"]
        for chunk in source_chunks
        if "id" in chunk
    ]

# Step 32 - append_source_references
def append_source_references(answer_text, source_chunks):
    ids = track_source_chunk_ids(source_chunks)
    return answer_text + "\nSources: [" + ", ".join(ids) + "]"

# Step 33 - query_rewrite
def query_rewrite(raw_query):
    import re

    # Reuse normalize_text() for Unicode normalization and whitespace cleanup.
    query = normalize_text(raw_query).lower()

    # Remove common conversational filler prefixes, including combinations.
    filler_patterns = [
        r"^(?:please\s+)+",
        r"^(?:(?:please\s+)?(?:could|can)\s+you\s+)+",
        r"^(?:(?:please\s+)?tell\s+me\s+)+",
        r"^(?:(?:please\s+)?i\s+want\s+to\s+know\s+)+",
    ]

    for pattern in filler_patterns:
        query = re.sub(pattern, "", query)

    # Repeatedly remove filler prefixes to handle combinations such as
    # "please tell me" and "could you please".
    previous = None
    while query != previous:
        previous = query
        for pattern in filler_patterns:
            query = re.sub(pattern, "", query)

    # Remove trailing sentence/question punctuation.
    query = re.sub(r"[?!.]+$", "", query).strip()

    return query

# Step 34 - hyde_retrieve
def hyde_retrieve(query, hypothetical_answer, chunks, embeddings, embed_model, k=5):
    # Embed the hypothetical answer instead of the original query.
    hypothetical_vector = embed_text(embed_model, hypothetical_answer)

    # Score the hypothetical answer against all chunk embeddings.
    scores = cosine_similarity_search(hypothetical_vector, embeddings)

    # Select the top-k chunks in descending similarity order.
    indices = top_k_indices(scores, k)

    return [chunks[i] for i in indices]

# Step 35 - reciprocal_rank_fusion
def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = {}

    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    return sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0])
    )

# Step 36 - bm25_search
def bm25_search(query, chunks, k=5, k1=1.5, b=0.75):
    import math
    from collections import Counter

    query_terms = query.lower().split()
    documents = [
        chunk["text"].lower().split()
        for chunk in chunks
    ]

    if not documents or not query_terms:
        return []

    n_docs = len(documents)
    doc_lengths = [len(doc) for doc in documents]
    avgdl = sum(doc_lengths) / n_docs if n_docs else 0.0

    # Document frequency for each unique query term.
    df = {}
    for term in set(query_terms):
        df[term] = sum(1 for doc in documents if term in doc)

    # Standard BM25 IDF variant specified in the prompt.
    idf = {
        term: math.log(
            (n_docs - freq + 0.5) / (freq + 0.5) + 1.0
        )
        for term, freq in df.items()
    }

    results = []

    for i, doc in enumerate(documents):
        if not doc:
            continue

        term_freqs = Counter(doc)
        score = 0.0

        for term in query_terms:
            if term not in term_freqs:
                continue

            tf = term_freqs[term]
            denominator = (
                tf
                + k1 * (1.0 - b + b * doc_lengths[i] / avgdl)
            )

            score += idf[term] * (
                tf * (k1 + 1.0)
            ) / denominator

        if score > 0.0:
            results.append((i, score))

    results.sort(key=lambda x: (-x[1], x[0]))

    return results[:k]

# Step 37 - hybrid_search (not yet solved)
# TODO: implement

# Step 38 - rerank_cross_encoder (not yet solved)
# TODO: implement

# Step 39 - maximal_marginal_relevance (not yet solved)
# TODO: implement

# Step 40 - filter_by_metadata (not yet solved)
# TODO: implement

# Step 41 - build_eval_set (not yet solved)
# TODO: implement

# Step 42 - hit_rate_at_k (not yet solved)
# TODO: implement

# Step 43 - recall_at_k (not yet solved)
# TODO: implement

# Step 44 - mean_reciprocal_rank (not yet solved)
# TODO: implement

# Step 45 - faithfulness_score (not yet solved)
# TODO: implement

# Step 46 - relevance_score (not yet solved)
# TODO: implement

# Step 47 - handle_no_context (not yet solved)
# TODO: implement

# Step 48 - deduplicate_chunks (not yet solved)
# TODO: implement

# Step 49 - cache_query_embedding (not yet solved)
# TODO: implement

# Step 50 - update_chat_memory (not yet solved)
# TODO: implement

# Step 51 - rewrite_followup (not yet solved)
# TODO: implement

