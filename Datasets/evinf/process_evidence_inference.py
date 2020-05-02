import sys, random, os, csv
from collections import namedtuple, defaultdict, Counter

import classes

sys.path.append("..")

sys.path.append(os.path.join("evidence-inference", "evidence_inference", "preprocess"))
import preprocessor


def extract_raw_abstract(article):
    return article.get_abstract(False).replace("<p>", "")


def init_doc(pmcid, abst_only):
    article = preprocessor.get_article(pmcid)
    if abst_only:
        # gotta add the same gunk as the preprocessor so it all lines up
        text = "TITLE:\n{}\n\n\n\n{}".format(article.get_title(), extract_raw_abstract(article))
    else:
        text = preprocessor.extract_raw_text(article)
    doc = classes.Doc.init_from_text(pmcid, text)
    return doc


def read_docs(abst_only=False):
    Prompt = namedtuple("Prompt", "i c o")
    docs = {}
    prompts = {}

    print("Reading prompts + articles")
    for prompt in preprocessor.read_prompts().to_dict("records"):
        pmcid = prompt["PMCID"]
        if pmcid not in docs:
            docs[pmcid] = init_doc(pmcid, abst_only)

        pid = prompt["PromptID"]
        if pid not in prompts:
            prompts[pid] = Prompt(prompt["Intervention"], prompt["Comparator"], prompt["Outcome"])

    print(len(docs))
    print(len(prompts))

    n_anns = 0
    n_bad_offsets = 0
    print("Processing annotations")
    anns = preprocessor.read_annotations().to_dict("records")
    for ann in anns:
        if abst_only and not ann["In Abstract"]:
            continue
        if not ann["Annotations"]:
            continue

        ev = classes.Span(ann["Evidence Start"], ann["Evidence End"], ann["Annotations"])
        label = ann["Label"]
        doc = docs[ann["PMCID"]]
        prompt = prompts[ann["PromptID"]]

        if doc.text[ev.i : ev.f] != ev.text:
            n_bad_offsets += 1
            continue

        n_anns += 1
        frame = classes.Frame(prompt.i.strip(), prompt.c.strip(), prompt.o.strip(), ev, label)
        doc.frames.append(frame)

    pmcids_docs = list(docs.items())
    for pmcid, doc in pmcids_docs:
        if not doc.frames:
            del docs[pmcid]

    print("Retained {}/{} valid annotations ({} w/ bad offsets)".format(n_anns, len(anns), n_bad_offsets))
    print("Retained {}/{} docs with nonzero prompts".format(len(docs), len(pmcids_docs)))

    return list(docs.values())

