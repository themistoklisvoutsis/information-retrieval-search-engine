# Local evaluation collection

Place the original document collection here before running the experiment. This folder is intentionally not bundled with the repository because the dataset was not present in the supplied archive and its redistribution terms were not verified.

Expected paths:

- `collection/docs/` — documents named by their IDs (the script expects IDs like `00001`).
- `collection/Queries.txt` — one query per non-empty line.
- `collection/Relevant.txt` — relevant document IDs per query, space separated, in corresponding order.

The script reads these relative to the repository root; launch `python main.py` from that directory. Do not commit the original dataset until you have permission to redistribute it.
