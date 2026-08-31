# RAG Pipeline

Construct a complete RAG system step by step: raw document ingestion and chunking, embeddings, dense and hybrid retrieval, prompt assembly, local generation, evaluation, and conversational memory. By the end you will have a pipeline that answers grounded questions with citations and can be measured on retrieval and answer-quality metrics.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** load_text_file
- [x] **2.** load_text_directory
- [x] **3.** extract_text_from_html
- [x] **4.** normalize_text
- [x] **5.** make_document
- [x] **6.** chunk_fixed_size
- [x] **7.** chunk_by_tokens
- [x] **8.** chunk_by_sentences
- [x] **9.** chunk_with_overlap
- [x] **10.** attach_chunk_metadata
- [x] **11.** load_embedding_model
- [x] **12.** embed_text
- [x] **13.** embed_chunks
- [ ] **14.** l2_normalize
- [ ] **15.** save_corpus
- [ ] **16.** cosine_similarity_search
- [ ] **17.** top_k_indices
- [ ] **18.** top_k_chunks
- [ ] **19.** retrieve
- [ ] **20.** build_faiss_index
- [ ] **21.** faiss_search
- [ ] **22.** compare_faiss_to_numpy
- [ ] **23.** save_faiss_index
- [ ] **24.** build_prompt_template
- [ ] **25.** format_context
- [ ] **26.** truncate_context
- [ ] **27.** add_system_instruction
- [ ] **28.** load_generator
- [ ] **29.** generate_answer
- [ ] **30.** rag_answer
- [ ] **31.** track_source_chunk_ids
- [ ] **32.** append_source_references
- [ ] **33.** query_rewrite
- [ ] **34.** hyde_retrieve
- [ ] **35.** reciprocal_rank_fusion
- [ ] **36.** bm25_search
- [ ] **37.** hybrid_search
- [ ] **38.** rerank_cross_encoder
- [ ] **39.** maximal_marginal_relevance
- [ ] **40.** filter_by_metadata
- [ ] **41.** build_eval_set
- [ ] **42.** hit_rate_at_k
- [ ] **43.** recall_at_k
- [ ] **44.** mean_reciprocal_rank
- [ ] **45.** faithfulness_score
- [ ] **46.** relevance_score
- [ ] **47.** handle_no_context
- [ ] **48.** deduplicate_chunks
- [ ] **49.** cache_query_embedding
- [ ] **50.** update_chat_memory
- [ ] **51.** rewrite_followup

---

Built on Deep-ML.
