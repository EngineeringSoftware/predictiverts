from pathlib import Path
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from seutil import BashUtils, IOUtils
from pts.main import Macros
import re

RE_SUBTOKENIZE = re.compile(r"(?<=[_$])(?!$)|(?<!^)(?=[_$])|(?<=[a-z])(?=[A-Z0-9])|(?<=[A-Z])(?=[0-9]|[A-Z][a-z0-9])|(?<=[0-9])(?=[a-zA-Z])")


def is_identifier(token: str) -> bool:
    return len(token) > 0 and \
            (token[0].isalpha() or token[0] in "_$") and \
            all([c.isalnum() or c in "_$" for c in token])


def subtokenize(token: str) -> List[str]:
    """
    Subtokenizes an identifier name into subtokens, by CamelCase and snake_case.
    """
    # Only subtokenize identifier words (starts with letter _$, contains only alnum and _$)
    if is_identifier(token):
        return RE_SUBTOKENIZE.split(token)
    else:
        return [token]


def subtokenize_batch(tokens: List[str]) -> Tuple[List[str], List[int]]:
    """
    Subtokenizes list of tokens.
    :return a list of subtokens, and a list of pointers to the original token indices.
    """
    sub_tokens = []
    src_indices = []
    for i, token in enumerate(tokens):
        new_sub_tokens = subtokenize(token)
        sub_tokens += new_sub_tokens
        src_indices += [i] * len(new_sub_tokens)
    return sub_tokens, src_indices

def get_tf_idf_query_similarity(corpus, query):
    """
    vectorizer: TfIdfVectorizer model
    docs_tfidf: tfidf vectors for all docs
    query: query doc

    return: cosine similarity between query and all docs
    """
    vectorizer = TfidfVectorizer()
    docs_tfidf = vectorizer.fit_transform(corpus)
    query_tfidf = vectorizer.transform([query])
    cosineSimilarities = cosine_similarity(query_tfidf, docs_tfidf).flatten()
    return cosineSimilarities

def run_TFIDF_baseline(project: str):
    eval_data_file = Macros.eval_data_dir / "mutated-eval-data" / f"{project}-ag-qualifiedname.json"

    res = []
    evaldata = IOUtils.load(eval_data_file)
    for evalitem in evaldata:
        res_item = {}
        failed_test_list = evalitem["failed_test_list_qualified"]
        passed_test_list = evalitem["passed_test_list_qualified"]
        total_test_list = failed_test_list + passed_test_list
        labels = [1 for _ in range(len(failed_test_list))] + [0 for _ in range(len(passed_test_list))]
        changed_files = [file.lower().replace(".java", "").replace("src/main/java/", "").replace("src/test/java/", "").replace(r"/", " ") for file in evalitem["changed_files"] if file.endswith(".java")]
        #print("changed_files", changed_files)
        corpus = [" ".join(subtokenize_batch(test.split("."))[0]).lower() for test in total_test_list]
        query = ["".join(changed_file) for changed_file in changed_files]
        similarity = get_tf_idf_query_similarity(corpus, query).tolist()
        #print("similarity", similarity)
        ekstazi_labels = []
        starts_labels = []
        for test in evalitem["failed_test_list"] + evalitem["passed_test_list"]:
            if test in evalitem["ekstazi_test_list"]:
                ekstazi_labels.append(1)
            else:
                ekstazi_labels.append(0)

            if test in evalitem["starts_test_list"]:
                starts_labels.append(1)
            else:
                starts_labels.append(0)

        res_item["labels"] = labels
        #res_item["similarity_per_file"] = similarity_per_file
        res_item["prediction_scores"] = similarity
        res_item["commit"] = evalitem["commit"]
        res_item["Ekstazi_labels"] = ekstazi_labels
        res_item["STARTS_labels"] = starts_labels
        res.append(res_item)

    # output_dir = Macros.results_dir / "modelResults" / project.split('_')[1] / "TFIDFBaseline" / "results"
    output_dir = Macros.model_data_dir / "rank-model" / project.split('_')[1] / "TFIDFBaseline" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    IOUtils.dump(output_dir/"per-sha-result.json", res, IOUtils.Format.jsonPretty)
