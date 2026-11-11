from collections import defaultdict
import os
import re
from nltk.stem import PorterStemmer
import math
from collections import Counter 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt



# Corpus loading and preprocessing
def get_available_docs(docs_folder="collection/docs"):
    return{
        name for name in os.listdir(docs_folder)
        if os.path.isfile(os.path.join(docs_folder,name))
    }

def check_missinf_docs_from_relevant(docs_folder="collection/docs",relevant_path="collection/Relevant.txt"):
    available_docs= get_available_docs(docs_folder)
    missing_docs=set()

    with open(relevant_path,"r",encoding="utf-8")as f:
        for line in f:
            ids= line.strip().split()
            if not ids:
                continue

            for raw_id in ids:
                doc_id=raw_id.zfill(5)
                if doc_id not in available_docs:
                    missing_docs.add(doc_id)
    return missing_docs, available_docs
def load_docs(base_folder="collection"):
    """
    Γυρίζει:
      - doc_ids: λίστα με τα ονόματα αρχείων (π.χ. '00001', '00002', ...)
      - doc_texts: λίστα με τα raw κείμενα
    """
    docs_folder = os.path.join(base_folder, "docs")
    doc_ids = []
    doc_texts = []

    for filename in sorted(os.listdir(docs_folder)):
        file_path = os.path.join(docs_folder, filename)
        if not os.path.isfile(file_path):
            continue
        doc_ids.append(filename)
        with open(file_path, "r", encoding="utf-8") as f:
            doc_texts.append(f.read())

    return doc_ids, doc_texts


# Only evaluate documents that are present in the local corpus.
def filter_relevant_lists(docs_folder="collection/docs", relevant_path="collection/Relevant.txt"):
    missing_docs, available_docs= check_missinf_docs_from_relevant(docs_folder,relevant_path)
    filtered_relevant=[]

    with open (relevant_path,"r",encoding="utf-8")as f:
        for line in f:
            ids=line.strip().split()
            if not ids:
                continue

            keep= []
            for raw_id in ids:
                doc_id=raw_id.zfill(5)
                if doc_id in available_docs:
                    keep.append(doc_id)
            filtered_relevant.append(keep)

    return filtered_relevant, missing_docs

stemmer= PorterStemmer()


STOPWORDS={
    "a","an","and","are","as","at","be","but","by","for","from","has","have",
    "he","her","his","i","in","is","it","its","of","on","or","our","she",
    "that","the","their","them","there","these","they","this","to","was",
    "we","were","with","you","your"
}


def preprocess(text):
    text = text.lower()
    tokens = re.findall(r"\w+", text)
    tokens = [t for t in tokens if len(t)>=3 and t not in STOPWORDS]
    tokens = [stemmer.stem(t)for t in tokens]
    return tokens



# Store per-document term frequencies for sparse retrieval.
def build_inverted_index(base_folder="collection"):
    inverted_index = defaultdict(lambda: defaultdict(int))
    doc_length = {}
    doc_max_tf = {}
    docs_folder = os.path.join(base_folder, "docs")

    for filename in sorted(os.listdir(docs_folder)):
        file_path = os.path.join(docs_folder, filename)

        if os.path.isfile(file_path):
            doc_id = filename
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            tokens = preprocess(text)
            doc_length[doc_id] = len(tokens)


            tf_counter = Counter(tokens)
            doc_max_tf[doc_id] = max(tf_counter.values()) if tf_counter else 0

            for token, tf in tf_counter.items():
                inverted_index[token][doc_id] = tf

    N = len(doc_length)
    return inverted_index, doc_length, doc_max_tf, N


def print_N_3Terms(base_folder="collection", sample_terms=3, sample_postings=5):
    inverted_index, doc_length, doc_max_tf, N = build_inverted_index(base_folder)

    print("\n**** ΑΡΙΘΜΟΣ DOCS N:", N, "****")
    print("**** #TERMS στο index:", len(inverted_index), "****")

    terms = list(inverted_index.keys())[:sample_terms]
    print("**** ΠΑΡΑΔΕΙΓΜΑΤΑ ΟΡΩΝ (sample) ****\n", terms)

    for t in terms:
        posting_sample = list(inverted_index[t].items())[:sample_postings]
        print(f"- term='{t}' | df={len(inverted_index[t])} | sample postings:", posting_sample)



# Vector Space Model: log TF-IDF and augmented TF-IDF
def idf_log10(N,df):
    if df<=0:
        return 0.0
    return math.log10(N/df)
def weight_log_tfidf(tf,idf):
    if tf<=0:
        return 0.0
    return (1.0+math.log10(tf))*idf

def weight_augmented_tfidf(tf,max_tf,idf):
    if tf<= 0 or max_tf<=0:
        return 0.0
    return (0.5+0.5 *(tf/max_tf))*idf
def compute_doc_norms(inverted_index, N, doc_max_tf=None, scheme="log_tfidf"):
    """
    scheme:
      - "log_tfidf" (classic)
      - "aug_tfidf" (augmented)
    """
    doc_norms = defaultdict(float)

    for term, postings in inverted_index.items():
        df = len(postings)
        if df == 0:
            continue

        idf = idf_log10(N, df)

        for doc_id, tf in postings.items():
            if scheme == "log_tfidf":
                w_d = weight_log_tfidf(tf, idf)

            elif scheme == "aug_tfidf":
                if doc_max_tf is None:
                    raise ValueError("doc_max_tf is required for scheme='aug_tfidf'")
                w_d = weight_augmented_tfidf(tf, doc_max_tf.get(doc_id, 0), idf)

            else:
                raise ValueError(f"Unknown scheme: {scheme}")

            doc_norms[doc_id] += w_d * w_d

    for doc_id in doc_norms:
        doc_norms[doc_id] = math.sqrt(doc_norms[doc_id])

    return doc_norms


def load_queries(queries_path="collection/Queries.txt"):
    queries=[]
    with open(queries_path,"r", encoding="utf-8")as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            queries.append(line)
    return queries

def preprocess_query(query_text):
    return preprocess(query_text)
def rank_query(query_text, inverted_index, N, doc_norms, doc_max_tf=None, top_k=10, scheme="log_tfidf"):
    tokens = preprocess_query(query_text)
    if not tokens:
        return []

    tf_q = Counter(tokens)
    max_tf_q = max(tf_q.values()) if tf_q else 0

    query_weights = {}
    for term, tf in tf_q.items():
        if term not in inverted_index:
            continue

        df = len(inverted_index[term])
        if df == 0:
            continue

        idf = idf_log10(N, df)

        if scheme == "log_tfidf":
            w_q = weight_log_tfidf(tf, idf)
        elif scheme == "aug_tfidf":
            w_q = weight_augmented_tfidf(tf, max_tf_q, idf)
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

        if w_q != 0.0:
            query_weights[term] = w_q

    if not query_weights:
        return []

    q_norm = math.sqrt(sum(w * w for w in query_weights.values()))
    if q_norm == 0.0:
        return []

    scores = defaultdict(float)

    for term, w_q in query_weights.items():
        postings = inverted_index.get(term, {})
        df = len(postings)
        if df == 0:
            continue

        idf = idf_log10(N, df)

        for doc_id, tf_d in postings.items():
            if scheme == "log_tfidf":
                w_d = weight_log_tfidf(tf_d, idf)

            elif scheme == "aug_tfidf":
                if doc_max_tf is None:
                    raise ValueError("doc_max_tf is required for scheme='aug_tfidf'")
                w_d = weight_augmented_tfidf(tf_d, doc_max_tf.get(doc_id, 0), idf)

            scores[doc_id] += w_q * w_d

    final_scores = []
    for doc_id, dot in scores.items():
        d_norm = doc_norms.get(doc_id, 0.0)
        if d_norm == 0.0:
            continue
        cos_sim = dot / (d_norm * q_norm)
        final_scores.append((doc_id, cos_sim))

    final_scores.sort(key=lambda x: x[1], reverse=True)
    return final_scores[:top_k]



# Retrieval metrics and model comparison
def evaluate_vsm_model(inverted_index, N, doc_norms, queries, filtered_relevant, top_k=10):

    t1 = time.time()

    precisions, recalls, f1s = [], [], []
    p_at_5_list, p_at_10_list = [], []

    for i, q in enumerate(queries):
        results = rank_query(q, inverted_index, N, doc_norms, top_k=top_k)
        ranked_docs = [doc_id for (doc_id, _) in results]
        relevant_docs = filtered_relevant[i]

        p = precision(ranked_docs, relevant_docs)
        r = recall(ranked_docs, relevant_docs)
        f = f1_score(p, r)

        p5 = precision_at_k(ranked_docs, relevant_docs, 5)
        p10 = precision_at_k(ranked_docs, relevant_docs, 10)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        p_at_5_list.append(p5)
        p_at_10_list.append(p10)

    retrieval_time = time.time() - t1

    return {
        "mean_precision": float(np.mean(precisions)),
        "mean_recall": float(np.mean(recalls)),
        "mean_f1": float(np.mean(f1s)),
        "mean_p_at_5": float(np.mean(p_at_5_list)),
        "mean_p_at_10": float(np.mean(p_at_10_list)),
        "retrieval_time": float(retrieval_time),
    }
def compare_q2_vs_q4(best_tfidf_params, top_k=10):

    filtered_relevant, missing_docs = filter_relevant_lists()
    doc_ids, doc_texts = load_docs("collection")
    queries = load_queries("collection/Queries.txt")


    t0 = time.time()
    inverted_index, doc_length, doc_max_tf, N = build_inverted_index("collection")
    doc_norms = compute_doc_norms(inverted_index, N, scheme="log_tfidf")

    vsm_index_time = time.time() - t0

    vsm_results = evaluate_vsm_model(
        inverted_index=inverted_index,
        N=N,
        doc_norms=doc_norms,
        queries=queries,
        filtered_relevant=filtered_relevant,
        top_k=top_k
    )


    tfidf_vectorizer = TfidfVectorizer(**best_tfidf_params)

    tfidf_results = evaluate_tfidf_model(
        vectorizer=tfidf_vectorizer,
        doc_ids=doc_ids,
        doc_texts=doc_texts,
        queries=queries,
        filtered_relevant=filtered_relevant,
        top_k=top_k
    )


    print("\n================ ΣΥΓΚΡΙΣΗ ΕΡ.2 (VSM) vs ΕΡ.4 (Best TF-IDF) ================")
    print(f"Evaluation on top_k={top_k} for {len(queries)} queries, {len(doc_ids)} docs")

    print("\n--- VSM (Ερ.2) ---")
    print(f"Mean Precision: {vsm_results['mean_precision']:.4f}")
    print(f"Mean Recall:    {vsm_results['mean_recall']:.4f}")
    print(f"Mean F1:        {vsm_results['mean_f1']:.4f}")
    print(f"Mean P@5:       {vsm_results['mean_p_at_5']:.4f}")
    print(f"Mean P@10:      {vsm_results['mean_p_at_10']:.4f}")
    print(f"Indexing time:  {vsm_index_time:.3f}s")
    print(f"Retrieval time: {vsm_results['retrieval_time']:.3f}s")

    print("\n--- Best TF-IDF (Ερ.4) ---")
    print(f"Mean Precision: {tfidf_results['mean_precision']:.4f}")
    print(f"Mean Recall:    {tfidf_results['mean_recall']:.4f}")
    print(f"Mean F1:        {tfidf_results['mean_f1']:.4f}")
    print(f"Mean P@5:       {tfidf_results['mean_p_at_5']:.4f}")
    print(f"Mean P@10:      {tfidf_results['mean_p_at_10']:.4f}")
    print(f"Indexing time:  {tfidf_results['index_time']:.3f}s")
    print(f"Retrieval time: {tfidf_results['retrieval_time']:.3f}s")

    return vsm_results, tfidf_results, vsm_index_time

import matplotlib.pyplot as plt

def precision(retrieved_docs, relevant_docs):
    """
    Precision = (# relevant retrieved) / (# retrieved)
    """
    if not retrieved_docs:
        return 0.0
    relevant_set = set(relevant_docs)
    hits = sum(1 for d in retrieved_docs if d in relevant_set)
    return hits / len(retrieved_docs)


def recall(retrieved_docs, relevant_docs):
    """
    Recall = (# relevant retrieved) / (# relevant total)
    """
    relevant_set = set(relevant_docs)
    if not relevant_set:
        return 0.0
    hits = sum(1 for d in retrieved_docs if d in relevant_set)
    return hits / len(relevant_set)


def f1_score(prec, rec):
    """
    F1 = 2PR / (P + R)
    """
    denom = prec + rec
    if denom == 0.0:
        return 0.0
    return 2.0 * prec * rec / denom


def precision_at_k(retrieved_docs, relevant_docs, k):
    """
    Precision@k: precision στα πρώτα k αποτελέσματα.
    """
    if k <= 0 or not retrieved_docs:
        return 0.0

    top_k = retrieved_docs[:k]
    relevant_set = set(relevant_docs)
    hits = sum(1 for d in top_k if d in relevant_set)

    denom = min(k, len(retrieved_docs))
    return hits / denom

def precision_recall_points(ranked_docs, relevant_docs):
    relevant_set = set(relevant_docs)
    total_relevant = len(relevant_set)
    n = len(ranked_docs)

    if n == 0:
        return [], []

    if total_relevant == 0:
        return [0.0] * n, [0.0] * n

    recall_vals = []
    precision_vals = []
    hits = 0

    for k, doc_id in enumerate(ranked_docs, start=1):
        if doc_id in relevant_set:
            hits += 1
        precision_vals.append(hits / k)
        recall_vals.append(hits / total_relevant)

    return recall_vals, precision_vals


def plot_precision_recall_curve(ranked_docs, relevant_docs, title="Precision–Recall Curve"):

    import matplotlib.pyplot as plt

    recalls, precisions = precision_recall_points(ranked_docs, relevant_docs)

    plt.figure()
    plt.plot(recalls, precisions, marker='o')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)


    plt.draw()
    plt.pause(0.001)


    return recalls, precisions



# scikit-learn TF-IDF evaluation
def evaluate_tfidf_model(
    vectorizer,
    doc_ids,
    doc_texts,
    queries,
    filtered_relevant,
    top_k=10,
):
    """
    Κάνει:
      - fit_transform στα docs (indexing)
      - ranking για κάθε query (cosine similarity)
      - υπολογισμό μετρικών (P, R, F1, P@5, P@10) με βάση τα top_k retrieved
      - επιστρέφει ΚΑΙ χρόνους (index/retrieval)
    """


    t0 = time.time()
    X_docs = vectorizer.fit_transform(doc_texts)
    index_time = time.time() - t0


    t1 = time.time()

    precisions, recalls, f1s = [], [], []
    p_at_5_list, p_at_10_list = [], []

    for i, q in enumerate(queries):
        X_q = vectorizer.transform([q])
        sims = cosine_similarity(X_q, X_docs)[0]

        ranked_idx = np.argsort(-sims)
        ranked_docs_all = [doc_ids[j] for j in ranked_idx]

        relevant_docs = filtered_relevant[i]


        retrieved = ranked_docs_all[:top_k]

        p = precision(retrieved, relevant_docs)
        r = recall(retrieved, relevant_docs)
        f = f1_score(p, r)

        p5 = precision_at_k(retrieved, relevant_docs, 5)
        p10 = precision_at_k(retrieved, relevant_docs, 10)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        p_at_5_list.append(p5)
        p_at_10_list.append(p10)

    retrieval_time = time.time() - t1

    return {
        "mean_precision": float(np.mean(precisions)),
        "mean_recall": float(np.mean(recalls)),
        "mean_f1": float(np.mean(f1s)),
        "mean_p_at_5": float(np.mean(p_at_5_list)),
        "mean_p_at_10": float(np.mean(p_at_10_list)),
        "index_time": float(index_time),
        "retrieval_time": float(retrieval_time),
    }



# Hyperparameter exploration (48 configurations in this version)
def run_experiments(best_by="mean_p_at_10", top_k_eval=10, print_top_n=5):
 

    filtered_relevant, missing_docs = filter_relevant_lists()
    print("Missing Docs:", len(missing_docs))

    doc_ids, doc_texts = load_docs("collection")
    queries = load_queries("collection/Queries.txt")
    print("Docs:", len(doc_ids), "Queries:", len(queries))

   
    ngram_ranges = [(1, 1), (1, 2)]
    use_idf_opts = [True, False]
    sublinear_tf_opts = [True]
    norm_opts = ["l1", "l2", None]
    min_df_opts = [1, 2]
    max_df_opts = [1.0, 0.85]

    experiments = []
    exp_id = 0

    for ngram in ngram_ranges:
        for use_idf in use_idf_opts:
            for sub_tf in sublinear_tf_opts:
                for norm in norm_opts:
                    for min_df in min_df_opts:
                        for max_df in max_df_opts:
                            exp_id += 1

                            vectorizer = TfidfVectorizer(
                                lowercase=True,
                                stop_words="english",
                                ngram_range=ngram,
                                use_idf=use_idf,
                                sublinear_tf=sub_tf,
                                norm=norm,
                                min_df=min_df,
                                max_df=max_df,
                            )

                            results = evaluate_tfidf_model(
                                vectorizer=vectorizer,
                                doc_ids=doc_ids,
                                doc_texts=doc_texts,
                                queries=queries,
                                filtered_relevant=filtered_relevant,
                                top_k=top_k_eval,
                            )

                            experiments.append({
                                "exp_id": exp_id,
                                "ngram_range": ngram,
                                "use_idf": use_idf,
                                "sublinear_tf": sub_tf,
                                "norm": norm,
                                "min_df": min_df,
                                "max_df": max_df,
                                **results
                            })


                            print(
                                f"[Exp {exp_id:03d}] ngram={ngram}, use_idf={use_idf}, sub_tf={sub_tf}, "
                                f"norm={norm}, min_df={min_df}, max_df={max_df} | "
                                f"P@10={results['mean_p_at_10']:.4f} F1={results['mean_f1']:.4f} "
                                f"(index {results['index_time']:.2f}s, retr {results['retrieval_time']:.2f}s)"
                            )

    if not experiments:
        print("No experiments ran.")
        return None, []

    if best_by not in experiments[0]:
        raise ValueError(
            f"best_by='{best_by}' δεν υπάρχει. Διαθέσιμα keys: {list(experiments[0].keys())}"
        )

    experiments_sorted = sorted(experiments, key=lambda e: e[best_by], reverse=True)


    print(f"\n================ TOP {print_top_n} configs (sorted by {best_by}) ================")
    for row in experiments_sorted[:print_top_n]:
        print(
            f"Exp {row['exp_id']:03d} | ngram={row['ngram_range']} use_idf={row['use_idf']} "
            f"sub_tf={row['sublinear_tf']} norm={row['norm']} min_df={row['min_df']} max_df={row['max_df']} | "
            f"P={row['mean_precision']:.4f} R={row['mean_recall']:.4f} F1={row['mean_f1']:.4f} "
            f"P@5={row['mean_p_at_5']:.4f} P@10={row['mean_p_at_10']:.4f} | "
            f"index={row['index_time']:.2f}s retr={row['retrieval_time']:.2f}s"
        )


    best = experiments_sorted[0]
    print("\n================ ΚΑΛΥΤΕΡΟ ΜΟΝΤΕΛΟ (ΕΡ.4) ================")
    print(f"Κριτήριο επιλογής: {best_by}")
    print(
        f"Experiment {best['exp_id']}: "
        f"ngram_range={best['ngram_range']}, use_idf={best['use_idf']}, "
        f"sublinear_tf={best['sublinear_tf']}, norm={best['norm']}, "
        f"min_df={best['min_df']}, max_df={best['max_df']}"
    )
    print("---- Metrics ----")
    print(f"Mean Precision: {best['mean_precision']:.4f}")
    print(f"Mean Recall:    {best['mean_recall']:.4f}")
    print(f"Mean F1:        {best['mean_f1']:.4f}")
    print(f"Mean P@5:       {best['mean_p_at_5']:.4f}")
    print(f"Mean P@10:      {best['mean_p_at_10']:.4f}")
    print("---- Times ----")
    print(f"Indexing time:  {best['index_time']:.3f}s")
    print(f"Retrieval time: {best['retrieval_time']:.3f}s")

    return best, experiments_sorted


from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD

def build_best_tfidf_vectorizer_from_best(best_row):
    """
    Φτιάχνει TfidfVectorizer από το best experiment που γύρισε το run_experiments().
    """
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=best_row["ngram_range"],
        use_idf=best_row["use_idf"],
        sublinear_tf=best_row["sublinear_tf"],
        norm=best_row["norm"],
        min_df=best_row["min_df"],
        max_df=best_row["max_df"],
    )

def pick_best_k_by_silhouette(X, k_values, random_state=42):
    """
    Δοκιμάζει πολλά k και διαλέγει το k με το μεγαλύτερο silhouette.
    Επιστρέφει: best_k, best_score, scores_dict
    """
    scores = {}
    best_k = None
    best_score = -999.0

    for k in k_values:
        if k <= 1:
            continue

        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)


        s = silhouette_score(X, labels)
        scores[k] = float(s)

        if s > best_score:
            best_score = s
            best_k = k

    return best_k, float(best_score), scores

def plot_elbow_inertia(X, k_values, title="Elbow (Inertia)"):
    """
    Προαιρετικό: elbow plot με inertia (SSE).
    """
    inertias = []
    for k in k_values:
        if k <= 1:
            continue
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    plt.figure()
    plt.plot([k for k in k_values if k > 1], inertias, marker="o")
    plt.title(title)
    plt.xlabel("k")
    plt.ylabel("Inertia (SSE)")
    plt.grid(True)
    plt.show()

def plot_2d_scatter(X_2d, labels, title="PCA 2D Scatter"):
    """
    Scatter plot σε 2D με labels.
    """
    plt.figure()
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, s=18)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.show()


# Clustering: sparse TF-IDF and dense sentence embeddings
def clustering_tfidf_bestk(doc_texts, tfidf_vectorizer, k_values=range(2, 21), do_elbow=True):
   
    print("\n[Q5] Clustering με TF–IDF (best από Ερ.4)...")


    X_tfidf = tfidf_vectorizer.fit_transform(doc_texts)


    if do_elbow:
        plot_elbow_inertia(X_tfidf, k_values, title="TF–IDF Elbow (Inertia)")


    best_k, best_sil, sil_scores = pick_best_k_by_silhouette(X_tfidf, k_values)
    print(f"[TF–IDF] Best k={best_k} (by silhouette), silhouette={best_sil:.4f}")


    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_tfidf)


    svd2 = TruncatedSVD(n_components=2, random_state=42)
    X_2d = svd2.fit_transform(X_tfidf)

    plot_2d_scatter(X_2d, labels, title=f"TF–IDF (SVD 2D) | k={best_k} | sil={best_sil:.4f}")

    return best_k, best_sil, sil_scores, X_2d, labels

def clustering_embeddings_bestk(doc_texts, k_values=range(2, 21), do_elbow=True):
  
    print("\n[Q5] Clustering με Embeddings (sentence-transformers)...")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    X_emb = model.encode(doc_texts, show_progress_bar=True)


    X_emb = normalize(X_emb)


    if do_elbow:
        plot_elbow_inertia(X_emb, k_values, title="Embeddings Elbow (Inertia)")


    best_k, best_sil, sil_scores = pick_best_k_by_silhouette(X_emb, k_values)
    print(f"[Embeddings] Best k={best_k} (by silhouette), silhouette={best_sil:.4f}")


    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_emb)


    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_emb)

    plot_2d_scatter(X_2d, labels, title=f"Embeddings (PCA 2D) | k={best_k} | sil={best_sil:.4f}")

    return best_k, best_sil, sil_scores, X_2d, labels



# End-to-end experiment runner
def main():

    filtered_relevant, missing_docs = filter_relevant_lists()

    print("**** Number of Missing Docs:", len(missing_docs), "****")
    if missing_docs:
        print("**** Missing Docs ****\n", sorted(missing_docs))

    print("Relevant docs στο Query 1 (μετά το φιλτράρισμα):", len(filtered_relevant[0]))


    inverted_index, doc_length, doc_max_tf, N = build_inverted_index("collection")


    print_N_3Terms("collection")


    doc_norms_log = compute_doc_norms(inverted_index, N, scheme="log_tfidf")
    doc_norms_aug = compute_doc_norms(inverted_index, N, doc_max_tf=doc_max_tf, scheme="aug_tfidf")


    queries = load_queries("collection/Queries.txt")
    print("Σύνολο queries:", len(queries))


    m = min(len(queries), len(filtered_relevant))

    print("\n=== ΕΡΩΤΗΜΑ 2: VSM (2 term-weighting schemes) ===")

    for i in range(m):
        q = queries[i]
        relevant_for_query = filtered_relevant[i]

        print(f"\nQuery {i+1}: {q}")


        res_log = rank_query(
            q, inverted_index, N,
            doc_norms_log,
            doc_max_tf=None,
            top_k=N,
            scheme="log_tfidf"
        )


        SHOW_K = 10

        print("\n--- VSM log-tfidf (Top-10) ---")
        for rank, (doc_id, score) in enumerate(res_log[:SHOW_K], start=1):
            print(f"  {rank:2d}. Doc {doc_id}  Score={score:.4f}")


        ranked_docs_log = [doc_id for (doc_id, _) in res_log]
        p = precision(ranked_docs_log, relevant_for_query)
        r = recall(ranked_docs_log, relevant_for_query)
        f1 = f1_score(p, r)
        p_at_5 = precision_at_k(ranked_docs_log, relevant_for_query, k=5)
        p_at_10 = precision_at_k(ranked_docs_log, relevant_for_query, k=10)

        print(f"  [log] Precision: {p:.3f} | Recall: {r:.3f} | F1: {f1:.3f} | P@5: {p_at_5:.3f} | P@10: {p_at_10:.3f}")


        plot_precision_recall_curve(ranked_docs_log, relevant_for_query, title=f"PR Curve (log) - Query {i+1}")


        res_aug = rank_query(
            q, inverted_index, N,
            doc_norms_aug,
            doc_max_tf=doc_max_tf,
            top_k=N,
            scheme="aug_tfidf"
        )

        
        print("\n--- VSM augmented-tfidf ---")
        for rank, (doc_id, score) in enumerate(res_aug[:SHOW_K], start=1):
            print(f"  {rank:2d}. Doc {doc_id}  Score={score:.4f}")
        ranked_docs_aug = [doc_id for (doc_id, _) in res_aug]
        p = precision(ranked_docs_aug, relevant_for_query)
        r = recall(ranked_docs_aug, relevant_for_query)
        f1 = f1_score(p, r)
        p_at_5 = precision_at_k(ranked_docs_aug, relevant_for_query, k=5)
        p_at_10 = precision_at_k(ranked_docs_aug, relevant_for_query, k=10)

        print(f"  [aug] Precision: {p:.3f} | Recall: {r:.3f} | F1: {f1:.3f} | P@5: {p_at_5:.3f} | P@10: {p_at_10:.3f}")

        plot_precision_recall_curve(ranked_docs_aug, relevant_for_query, title=f"PR Curve (aug) - Query {i+1}")


    print("\n================ ΕΡΩΤΗΜΑ 4: TF-IDF ΠΕΙΡΑΜΑΤΑ ================")
    best, experiments_sorted = run_experiments(best_by="mean_p_at_10", top_k_eval=10, print_top_n=5)

    best_params = {
        "lowercase": True,
        "stop_words": "english",
        "ngram_range": best["ngram_range"],
        "use_idf": best["use_idf"],
        "sublinear_tf": best["sublinear_tf"],
        "norm": best["norm"],
        "min_df": best["min_df"],
        "max_df": best["max_df"],
    }

    compare_q2_vs_q4(best_params, top_k=10)

    print("\n================ ΕΡΩΤΗΜΑ 5: CLUSTERING (BEST-K + PLOTS) ================")
    doc_ids, doc_texts = load_docs("collection")

    best_vectorizer = build_best_tfidf_vectorizer_from_best(best)
    k_values = range(2, 21)

    tf_k, tf_sil, tf_sils, X_tfidf_2d, labels_tfidf = clustering_tfidf_bestk(
        doc_texts,
        tfidf_vectorizer=best_vectorizer,
        k_values=k_values,
        do_elbow=True
    )

    emb_k, emb_sil, emb_sils, X_emb_2d, labels_emb = clustering_embeddings_bestk(
        doc_texts,
        k_values=k_values,
        do_elbow=True
    )

    print("\n[Q5] Σύγκριση Silhouette scores (best k):")
    print(f"  TF–IDF:     k={tf_k}, silhouette={tf_sil:.4f}")
    print(f"  Embeddings: k={emb_k}, silhouette={emb_sil:.4f}")


if __name__ == "__main__":
        main()

