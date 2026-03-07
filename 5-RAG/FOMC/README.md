RAG FOMC Update (March 2nd, 2026)

Current status: 

The notebook runs end-to-end with Qwen 0.5B, builds a semantic RAG index over the FOMC corpus, and shows that semantic retrieval improves factual QA compared to no-retrieval and keyword baselines.


Remaining issues:

1. Statement-to-decision label alignment is still sensitive to date mismatches and depends on the chosen merge strategy (release date vs decision date).

2. Citation scoring shows 0% because model outputs rarely match the strict citation regex format, even when grounded.

3. Policy classification performance is weak: the model heavily defaults to predicting “Hold” for most statements, indicating class bias and insufficient signal extraction from short excerpts.

4. Differencing remains shallow and does not reliably link textual changes to policy shifts.
