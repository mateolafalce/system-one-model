"""Export public gold datasets to System One JSONL. English only."""

from __future__ import annotations

import random
from typing import Any, Callable, Iterator

import config  # noqa: F401
from schema import Label, Question, Record
from templates import (
    HELPSTEER_CRITERIA,
    SST5_CRITERIA,
    STAR_CRITERIA,
    TOXICITY_ORDINAL,
)

Builder = Callable[[int | None, int], list[Record]]


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _subsample(rows: list[Any], cap: int | None, seed: int) -> list[Any]:
    if cap is None or cap >= len(rows):
        return rows
    return _rng(seed).sample(rows, cap)


# Script-based Hub datasets were removed in `datasets` 5. Try parquet mirrors first.
HF_ID_FALLBACKS: dict[str, list[str]] = {
    "PolyAI/banking77": ["PolyAI/banking77", "mteb/banking77"],
    "clinc_oos": ["clinc_oos", "SetFit/clinc_oos"],
    "SetFit/sst5": ["SetFit/sst5", "Marqo/sst5"],
    "ucirvine/sms_spam": ["ucirvine/sms_spam", "sms_spam"],
    "CogComp/trec": ["CogComp/trec", "SetFit/trec"],
}


def _load(hf_id: str, config: str | None = None, split: str | None = None):
    from datasets import load_dataset

    ids = HF_ID_FALLBACKS.get(hf_id, [hf_id])
    last_exc: Exception | None = None
    for cand in ids:
        try:
            kwargs: dict[str, Any] = {}
            if config is not None:
                ds = load_dataset(cand, config, **kwargs)
            else:
                ds = load_dataset(cand)
            if split is not None:
                return ds[split]
            return ds
        except Exception as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"failed to load {hf_id}: {last_exc}") from last_exc


def _assign_random_splits(recs: list[Record], seed: int) -> None:
    rng = _rng(seed)
    order = list(range(len(recs)))
    rng.shuffle(order)
    n = len(recs)
    n_train = int(0.90 * n)
    n_val = int(0.95 * n)
    for rank, i in enumerate(order):
        recs[i].split = "train" if rank < n_train else ("val" if rank < n_val else "test")


def _official_split_name(name: str) -> str:
    if name in {"train", "validation", "val", "test", "zeroshot"}:
        return "val" if name == "validation" else name
    return "train"


def _iter_named_splits(ds) -> Iterator[tuple[str, Any]]:
    if hasattr(ds, "keys"):
        for k in ds.keys():
            yield _official_split_name(k), ds[k]
    else:
        yield "train", ds


def _label_names(split) -> list[str] | None:
    feats = getattr(split, "features", None)
    if not feats:
        return None
    for key in ("label", "intent", "coarse_label", "fine_label"):
        if key in feats and hasattr(feats[key], "names"):
            return list(feats[key].names)
    return None


def _readable(name: str) -> str:
    return str(name).replace("_", " ").replace("-", " ").strip()


# Canonical BANKING77 names (dataset 5 often stores a plain int, no ClassLabel).
BANKING77_LABELS = [
    "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support",
    "automatic_top_up", "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
    "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival",
    "card_delivery_estimate", "card_linking", "card_not_working",
    "card_payment_fee_charged", "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate", "card_swallowed", "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised", "change_pin", "compromised_card",
    "contactless_not_working", "country_support", "declined_card_payment",
    "declined_cash_withdrawal", "declined_transfer", "direct_debit_payment_not_recognised",
    "disposable_card_limits", "edit_personal_details", "exchange_charge",
    "exchange_rate", "exchange_via_app", "extra_charge_on_statement", "failed_transfer",
    "fiat_currency_support", "get_disposable_virtual_card", "get_physical_card",
    "getting_spare_card", "getting_virtual_card", "lost_or_stolen_card",
    "lost_or_stolen_phone", "order_physical_card", "passcode_forgotten",
    "pending_card_payment", "pending_cash_withdrawal", "pending_top_up",
    "pending_transfer", "pin_blocked", "receiving_money", "Refund_not_showing_up",
    "request_refund", "reverted_card_payment?", "supported_cards_and_currencies",
    "terminate_account", "top_up_by_bank_transfer_charge", "top_up_by_card_charge",
    "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits", "top_up_reverted",
    "topping_up_by_card", "transaction_charged_twice", "transfer_fee_charged",
    "transfer_into_account", "transfer_not_received_by_recipient", "transfer_timing",
    "unable_to_verify_identity", "verify_my_identity", "verify_source_of_funds",
    "verify_top_up", "virtual_card_not_working", "visa_or_mastercard",
    "why_verify_identity", "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]


def _class_names(split, key: str, fallback: list[str] | None = None) -> list[str]:
    feat = split.features.get(key) if hasattr(split, "features") else None
    names = getattr(feat, "names", None)
    if names:
        return list(names)
    if fallback:
        return list(fallback)
    raise RuntimeError(f"no class names for {key} and no fallback")


def _decode_label(row: dict, key: str, names: list[str]) -> str:
    raw = row[key]
    if isinstance(raw, str):
        return raw
    return names[int(raw)]


# --- BANKING77 coarse buckets (plan section 4.1) ---
BANKING77_COARSE: dict[str, set[str]] = {
    "card": {
        "activate_my_card", "apple_pay_or_google_pay", "card_about_to_expire", "card_acceptance",
        "card_arrival", "card_delivery_estimate", "card_linking", "card_not_working",
        "card_swallowed", "change_pin", "contactless_not_working", "disposable_card_limits",
        "get_disposable_virtual_card", "get_physical_card", "getting_spare_card",
        "getting_virtual_card", "order_physical_card", "pin_blocked",
        "supported_cards_and_currencies", "virtual_card_not_working", "visa_or_mastercard",
        "declined_card_payment", "pending_card_payment", "reverted_card_payment?",
    },
    "transfer": {
        "cancel_transfer", "declined_transfer", "failed_transfer", "pending_transfer",
        "receiving_money", "transfer_into_account", "transfer_not_received_by_recipient",
        "transfer_timing", "balance_not_updated_after_bank_transfer", "beneficiary_not_allowed",
        "balance_not_updated_after_cheque_or_cash_deposit",
    },
    "identity": {
        "edit_personal_details", "passcode_forgotten", "unable_to_verify_identity",
        "verify_my_identity", "why_verify_identity", "age_limit",
    },
    "fee": {
        "card_payment_fee_charged", "cash_withdrawal_charge", "exchange_charge",
        "extra_charge_on_statement", "top_up_by_bank_transfer_charge", "top_up_by_card_charge",
        "transfer_fee_charged", "transaction_charged_twice",
    },
    "loan": {"verify_source_of_funds"},
    "app": {
        "atm_support", "country_support", "exchange_via_app", "fiat_currency_support",
        "automatic_top_up", "pending_top_up", "top_up_by_cash_or_cheque", "top_up_failed",
        "top_up_limits", "top_up_reverted", "topping_up_by_card", "exchange_rate",
    },
    "fraud": {
        "compromised_card", "lost_or_stolen_card", "lost_or_stolen_phone",
        "card_payment_not_recognised", "cash_withdrawal_not_recognised",
        "direct_debit_payment_not_recognised", "wrong_amount_of_cash_received",
        "wrong_exchange_rate_for_cash_withdrawal",
    },
    "other": set(),
}

COARSE_CRITERIA = {
    "card": "Cards, PIN, Apple/Google Pay, virtual or physical card issues",
    "transfer": "Sending, receiving, or pending transfers and deposits",
    "identity": "Identity, personal details, passcode, or verification",
    "fee": "Fees, charges, exchange fees, or double charges",
    "loan": "Source of funds, credit, or lending checks",
    "app": "App features, top-ups, ATM, country or currency support",
    "fraud": "Lost/stolen, unrecognized transactions, compromise",
    "other": "Anything else, including refunds and account closure",
}


def _coarse_bucket(label: str) -> str:
    for bucket, names in BANKING77_COARSE.items():
        if label in names:
            return bucket
    return "other"


def _choice_record(
    *,
    rec_id: str,
    group_id: str,
    source: str,
    split: str,
    family: str,
    state: Any,
    qid: str,
    instructions: str,
    criteria: dict[str, str],
    choice: str,
) -> Record:
    return Record(
        id=rec_id,
        group_id=group_id,
        source=source,
        split=split,
        state=state,
        question=Question(id=qid, type="choice", instructions=instructions, criteria=criteria),
        label=Label(kind="hard", choice=choice),
        family=family,
    )


def _noul_record(
    *,
    rec_id: str,
    group_id: str,
    source: str,
    split: str,
    family: str,
    state: Any,
    qid: str,
    instructions: str,
    noul: float,
) -> Record:
    return Record(
        id=rec_id,
        group_id=group_id,
        source=source,
        split=split,
        state=state,
        question=Question(id=qid, type="noul", instructions=instructions),
        label=Label(kind="hard", noul=noul),
        family=family,
    )


def _score_record(
    *,
    rec_id: str,
    group_id: str,
    source: str,
    split: str,
    family: str,
    state: Any,
    qid: str,
    instructions: str,
    criteria: list[str],
    score: int,
) -> Record:
    return Record(
        id=rec_id,
        group_id=group_id,
        source=source,
        split=split,
        state=state,
        question=Question(id=qid, type="score", instructions=instructions, criteria=criteria),
        label=Label(kind="hard", score=int(score)),
        family=family,
    )


def export_banking77(cap: int | None = None, seed: int = 0) -> list[Record]:
    ds = _load("PolyAI/banking77")
    out: list[Record] = []
    for split_name, split in _iter_named_splits(ds):
        names = _class_names(split, "label", BANKING77_LABELS)
        criteria = {n: f"The user wants: {_readable(n)}." for n in names}
        rows = list(split)
        if cap:
            rows = _subsample(rows, cap, seed + hash(split_name) % 1000)
        for i, row in enumerate(rows):
            lab = _decode_label(row, "label", names)
            gid = f"banking77-{split_name}-{i:05d}"
            state = {"text": row["text"]}
            out.append(
                _choice_record(
                    rec_id=f"{gid}-intent",
                    group_id=gid,
                    source="PolyAI/banking77",
                    split=split_name,
                    family="banking77",
                    state=state,
                    qid="intent",
                    instructions="Which banking intent matches this customer message?",
                    criteria=criteria,
                    choice=lab,
                )
            )
            out.append(
                _choice_record(
                    rec_id=f"{gid}-coarse",
                    group_id=gid,
                    source="PolyAI/banking77",
                    split=split_name,
                    family="banking77_coarse",
                    state=state,
                    qid="coarse",
                    instructions="Which department-style bucket fits this banking message?",
                    criteria=COARSE_CRITERIA,
                    choice=_coarse_bucket(lab),
                )
            )
    return out


def export_clinc(cap: int | None = None, seed: int = 0) -> list[Record]:
    ds = _load("clinc_oos", "plus")
    out: list[Record] = []
    for split_name, split in _iter_named_splits(ds):
        names = _class_names(split, "intent")
        criteria = {n: f"Intent: {_readable(n)}." for n in names}
        if "oos" not in criteria:
            criteria["oos"] = "Out of scope: none of the listed intents."
        rows = list(split)
        if cap:
            rows = _subsample(rows, cap, seed)
        for i, row in enumerate(rows):
            lab = names[int(row["intent"])]
            gid = f"clinc-{split_name}-{i:05d}"
            out.append(
                _choice_record(
                    rec_id=f"{gid}-intent",
                    group_id=gid,
                    source="clinc_oos",
                    split=split_name,
                    family="clinc",
                    state={"text": row["text"]},
                    qid="intent",
                    instructions="Which intent best matches this utterance? Include out-of-scope if none apply.",
                    criteria=criteria,
                    choice=lab,
                )
            )
    return out


def export_massive(cap: int | None = None, seed: int = 0) -> list[Record]:
    ds = _load("AmazonScience/massive")
    out: list[Record] = []
    for split_name, split in _iter_named_splits(ds):
        names = _class_names(split, "intent")
        criteria = {n: f"Intent: {_readable(n)}." for n in names}
        rows = [r for r in split if str(r.get("locale", "en-US")).startswith("en")]
        if cap:
            rows = _subsample(rows, cap, seed)
        for i, row in enumerate(rows):
            lab = names[int(row["intent"])]
            gid = f"massive-{split_name}-{i:05d}"
            out.append(
                _choice_record(
                    rec_id=f"{gid}-intent",
                    group_id=gid,
                    source="AmazonScience/massive",
                    split=split_name,
                    family="massive",
                    state={"text": row["utt"]},
                    qid="intent",
                    instructions="Which spoken-style intent matches this utterance?",
                    criteria=criteria,
                    choice=lab,
                )
            )
    return out


def export_ag_news(cap: int | None = 8000, seed: int = 0) -> list[Record]:
    ds = _load("fancyzhx/ag_news")
    names = ["world", "sports", "business", "sci_tech"]
    criteria = {
        "world": "World news, geopolitics, international events",
        "sports": "Sports",
        "business": "Business, markets, companies",
        "sci_tech": "Science or technology",
    }
    out: list[Record] = []
    for split_name, split in _iter_named_splits(ds):
        rows = list(split)
        if cap:
            rows = _subsample(rows, cap, seed)
        for i, row in enumerate(rows):
            lab = names[int(row["label"])]
            gid = f"agnews-{split_name}-{i:05d}"
            out.append(
                _choice_record(
                    rec_id=f"{gid}-topic",
                    group_id=gid,
                    source="fancyzhx/ag_news",
                    split=split_name,
                    family="ag_news",
                    state={"text": row["text"]},
                    qid="topic",
                    instructions="Which news topic is this article?",
                    criteria=criteria,
                    choice=lab,
                )
            )
    return out


def export_trec(cap: int | None = None, seed: int = 0) -> list[Record]:
    ds = _load("CogComp/trec")
    out: list[Record] = []
    for split_name, split in _iter_named_splits(ds):
        names = _class_names(split, "coarse_label")
        criteria = {n: f"Question type: {_readable(n)}." for n in names}
        rows = list(split)
        if cap:
            rows = _subsample(rows, cap, seed)
        for i, row in enumerate(rows):
            lab = names[int(row["coarse_label"])]
            gid = f"trec-{split_name}-{i:05d}"
            out.append(
                _choice_record(
                    rec_id=f"{gid}-coarse",
                    group_id=gid,
                    source="CogComp/trec",
                    split=split_name,
                    family="trec",
                    state={"text": row["text"]},
                    qid="coarse",
                    instructions="What type of answer does this question expect?",
                    criteria=criteria,
                    choice=lab,
                )
            )
    return out


def export_emotion(cap: int | None = 8000, seed: int = 0) -> list[Record]:
    ds = _load("dair-ai/emotion")
    out: list[Record] = []
    for split_name, split in _iter_named_splits(ds):
        names = _class_names(split, "label")
        criteria = {n: f"The writer expresses {_readable(n)}." for n in names}
        rows = list(split)
        if cap:
            rows = _subsample(rows, cap, seed)
        for i, row in enumerate(rows):
            lab = names[int(row["label"])]
            gid = f"emotion-{split_name}-{i:05d}"
            out.append(
                _choice_record(
                    rec_id=f"{gid}-label",
                    group_id=gid,
                    source="dair-ai/emotion",
                    split=split_name,
                    family="emotion",
                    state={"text": row["text"]},
                    qid="label",
                    instructions="Which emotion does this text express?",
                    criteria=criteria,
                    choice=lab,
                )
            )
    return out


def export_bitext(cap: int | None = 10000, seed: int = 0) -> list[Record]:
    ds = _load("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    split = ds["train"] if hasattr(ds, "keys") and "train" in ds else ds
    intents = sorted({str(r["intent"]) for r in split})
    criteria = {n: f"Support intent: {_readable(n)}." for n in intents}
    rows = list(split)
    if cap:
        rows = _subsample(rows, cap, seed)
    out: list[Record] = []
    for i, row in enumerate(rows):
        gid = f"bitext-train-{i:05d}"
        split_name = "train" if i < int(0.9 * len(rows)) else ("val" if i < int(0.95 * len(rows)) else "test")
        text = row.get("instruction") or row.get("utterance") or row.get("text") or ""
        out.append(
            _choice_record(
                rec_id=f"{gid}-intent",
                group_id=gid,
                source="bitext/Bitext-customer-support-llm-chatbot-training-dataset",
                split=split_name,
                family="bitext",
                state={"text": text},
                qid="intent",
                instructions="Which customer-support intent matches this chat?",
                criteria=criteria,
                choice=str(row["intent"]),
            )
        )
    return out


def export_tweet_eval(config: str, family: str, instructions: str, cap: int | None, seed: int, zeroshot: bool = False) -> list[Record]:
    ds = _load("cardiffnlp/tweet_eval", config)
    out: list[Record] = []
    for split_name, split in _iter_named_splits(ds):
        names = _class_names(split, "label")
        criteria = {n: f"Label: {_readable(n)}." for n in names}
        rows = list(split)
        if cap:
            rows = _subsample(rows, cap, seed)
        use_split = "zeroshot" if zeroshot else split_name
        for i, row in enumerate(rows):
            lab = names[int(row["label"])]
            gid = f"tweet_{config}-{split_name}-{i:05d}"
            out.append(
                _choice_record(
                    rec_id=f"{gid}-label",
                    group_id=gid,
                    source=f"cardiffnlp/tweet_eval:{config}",
                    split=use_split,
                    family=family,
                    state={"text": row["text"]},
                    qid="label",
                    instructions=instructions,
                    criteria=criteria,
                    choice=lab,
                )
            )
    return out


def export_tweet_emotion_zeroshot(cap: int | None = 6000, seed: int = 0) -> list[Record]:
    return export_tweet_eval("emotion", "tweet_emotion", "Which emotion does this tweet express?", cap, seed, zeroshot=True)


def export_tweet_offensive(cap: int | None = 6000, seed: int = 0) -> list[Record]:
    return export_tweet_eval("offensive", "tweet_offensive", "Is this tweet offensive or not?", cap, seed, zeroshot=False)


def export_boolq(cap: int | None = 10000, seed: int = 0) -> list[Record]:
    ds = _load("google/boolq")
    out: list[Record] = []
    for split_name, split in _iter_named_splits(ds):
        rows = list(split)
        if cap:
            rows = _subsample(rows, cap, seed)
        for i, row in enumerate(rows):
            ans = row["answer"]
            yes = 1.0 if ans in (True, "true", "yes", 1) else 0.0
            gid = f"boolq-{split_name}-{i:05d}"
            out.append(
                _noul_record(
                    rec_id=f"{gid}-yes",
                    group_id=gid,
                    source="google/boolq",
                    split=split_name,
                    family="boolq",
                    state={"question": row["question"], "passage": row["passage"]},
                    qid="yes",
                    instructions="The passage answers the question with yes.",
                    noul=yes,
                )
            )
    return out


def export_glue(config: str, family: str, instructions: str, pos_label: int, cap: int | None, seed: int, zeroshot: bool = False) -> list[Record]:
    ds = _load("nyu-mll/glue", config)
    out: list[Record] = []
    for split_name, split in _iter_named_splits(ds):
        if split_name == "test":
            # GLUE test has no labels.
            continue
        rows = [r for r in split if r.get("label", -1) != -1]
        if cap:
            rows = _subsample(rows, cap, seed)
        use_split = "zeroshot" if zeroshot else split_name
        for i, row in enumerate(rows):
            noul = 1.0 if int(row["label"]) == pos_label else 0.0
            if config == "rte":
                state = {"premise": row["sentence1"], "hypothesis": row["sentence2"]}
            elif config == "qnli":
                state = {"question": row["question"], "sentence": row["sentence"]}
            elif config == "cola":
                state = {"text": row["sentence"]}
            else:
                state = {"text": str(row)}
            gid = f"glue_{config}-{split_name}-{i:05d}"
            out.append(
                _noul_record(
                    rec_id=f"{gid}-label",
                    group_id=gid,
                    source=f"nyu-mll/glue:{config}",
                    split=use_split,
                    family=family,
                    state=state,
                    qid="label",
                    instructions=instructions,
                    noul=noul,
                )
            )
    return out


def export_rte(cap: int | None = None, seed: int = 0) -> list[Record]:
    return export_glue("rte", "rte", "The hypothesis follows from the premise.", pos_label=0, cap=cap, seed=seed)


def export_qnli(cap: int | None = 8000, seed: int = 0) -> list[Record]:
    return export_glue("qnli", "qnli", "The sentence contains the answer to the question.", pos_label=0, cap=cap, seed=seed)


def export_cola_zeroshot(cap: int | None = None, seed: int = 0) -> list[Record]:
    return export_glue("cola", "cola", "This sentence is grammatical.", pos_label=1, cap=cap, seed=seed, zeroshot=True)


def export_paws(cap: int | None = 8000, seed: int = 0) -> list[Record]:
    ds = _load("google-research-datasets/paws", "labeled_final")
    out: list[Record] = []
    for split_name, split in _iter_named_splits(ds):
        rows = list(split)
        if cap:
            rows = _subsample(rows, cap, seed)
        for i, row in enumerate(rows):
            gid = f"paws-{split_name}-{i:05d}"
            out.append(
                _noul_record(
                    rec_id=f"{gid}-paraphrase",
                    group_id=gid,
                    source="google-research-datasets/paws",
                    split=split_name,
                    family="paws",
                    state={"sentence1": row["sentence1"], "sentence2": row["sentence2"]},
                    qid="paraphrase",
                    instructions="The two sentences are paraphrases.",
                    noul=float(row["label"]),
                )
            )
    return out


def export_scitail(cap: int | None = 8000, seed: int = 0) -> list[Record]:
    try:
        ds = _load("allenai/scitail", "tsv_format")
    except Exception:
        ds = _load("allenai/scitail", "dgem_format")
    out: list[Record] = []
    for split_name, split in _iter_named_splits(ds):
        rows = list(split)
        if cap:
            rows = _subsample(rows, cap, seed)
        for i, row in enumerate(rows):
            label = row.get("label") or row.get("gold_label")
            yes = 1.0 if str(label).lower() in {"entails", "entailment", "1", "true"} else 0.0
            prem = row.get("premise") or row.get("sentence1")
            hyp = row.get("hypothesis") or row.get("sentence2")
            gid = f"scitail-{split_name}-{i:05d}"
            out.append(
                _noul_record(
                    rec_id=f"{gid}-entail",
                    group_id=gid,
                    source="allenai/scitail",
                    split=split_name,
                    family="scitail",
                    state={"premise": prem, "hypothesis": hyp},
                    qid="entail",
                    instructions="The hypothesis is entailed by the premise.",
                    noul=yes,
                )
            )
    return out


def export_enron(cap: int | None = 8000, seed: int = 0) -> list[Record]:
    ds = _load("SetFit/enron_spam")
    out: list[Record] = []
    for split_name, split in _iter_named_splits(ds):
        rows = list(split)
        if cap:
            rows = _subsample(rows, cap, seed)
        for i, row in enumerate(rows):
            text = row.get("text") or row.get("message") or row.get("email") or ""
            subj = row.get("subject") or ""
            noul = float(row["label"]) if not isinstance(row["label"], str) else (1.0 if "spam" in str(row["label"]).lower() else 0.0)
            gid = f"enron-{split_name}-{i:05d}"
            out.append(
                _noul_record(
                    rec_id=f"{gid}-spam",
                    group_id=gid,
                    source="SetFit/enron_spam",
                    split=split_name,
                    family="enron_spam",
                    state={"subject": subj, "body": text} if subj else {"text": text},
                    qid="spam",
                    instructions="This email is spam.",
                    noul=noul,
                )
            )
    return out


def export_sms(cap: int | None = None, seed: int = 0) -> list[Record]:
    try:
        ds = _load("ucirvine/sms_spam")
    except Exception:
        ds = _load("sms_spam")
    out: list[Record] = []
    for split_name, split in _iter_named_splits(ds):
        names = None
        if "label" in split.features and hasattr(split.features["label"], "names"):
            names = _class_names(split, "label")
        rows = list(split)
        if cap:
            rows = _subsample(rows, cap, seed)
        for i, row in enumerate(rows):
            lab = int(row["label"])
            if names:
                noul = 1.0 if "spam" in names[lab].lower() else 0.0
            else:
                noul = 1.0 if lab == 1 else 0.0
            gid = f"sms-{split_name}-{i:05d}"
            out.append(
                _noul_record(
                    rec_id=f"{gid}-spam",
                    group_id=gid,
                    source="ucirvine/sms_spam",
                    split=split_name,
                    family="sms_spam",
                    state={"text": row["sms"] if "sms" in row else row["text"]},
                    qid="spam",
                    instructions="This message is spam.",
                    noul=noul,
                )
            )
    if len({r.split for r in out}) == 1:
        _assign_random_splits(out, seed)
    return out


def export_civil_noul(cap: int | None = 8000, seed: int = 0) -> list[Record]:
    ds = _load("google/civil_comments")
    split = ds["train"]
    rng = _rng(seed)
    want = cap or 8000
    idx = rng.sample(range(len(split)), min(len(split), want * 6))
    pos, neg = [], []
    for i in idx:
        row = split[int(i)]
        (pos if float(row["toxicity"]) >= 0.5 else neg).append(row)
        if len(pos) >= want // 2 and len(neg) >= want // 2:
            break
    half = min(want // 2, len(pos), len(neg))
    picked = pos[:half] + neg[:half]
    rng.shuffle(picked)
    out: list[Record] = []
    for i, row in enumerate(picked):
        split_name = "train" if i < int(0.9 * len(picked)) else ("val" if i < int(0.95 * len(picked)) else "test")
        gid = f"civil-{i:05d}"
        out.append(
            _noul_record(
                rec_id=f"{gid}-toxic",
                group_id=gid,
                source="google/civil_comments",
                split=split_name,
                family="civil_noul",
                state={"text": row["text"]},
                qid="toxic",
                instructions="This comment is toxic.",
                noul=1.0 if float(row["toxicity"]) >= 0.5 else 0.0,
            )
        )
    return out


def export_civil_score(cap: int | None = 8000, seed: int = 0) -> list[Record]:
    ds = _load("google/civil_comments")
    split = ds["train"]
    rng = _rng(seed)
    want = cap or 8000
    idx = rng.sample(range(len(split)), min(len(split), want))
    rows = [split[int(i)] for i in idx]

    def bin_tox(v: float) -> int:
        if v < 0.2:
            return 0
        if v < 0.5:
            return 1
        if v < 0.8:
            return 2
        return 3

    out: list[Record] = []
    for i, row in enumerate(rows):
        split_name = "train" if i < int(0.9 * len(rows)) else ("val" if i < int(0.95 * len(rows)) else "test")
        gid = f"civil_ord-{i:05d}"
        out.append(
            _score_record(
                rec_id=f"{gid}-level",
                group_id=gid,
                source="google/civil_comments",
                split=split_name,
                family="civil_score",
                state={"text": row["text"]},
                qid="level",
                instructions="How toxic is this comment?",
                criteria=TOXICITY_ORDINAL,
                score=bin_tox(float(row["toxicity"])),
            )
        )
    return out


def export_prompt_injection(cap: int | None = None, seed: int = 0) -> list[Record]:
    ds = _load("deepset/prompt-injections")
    out: list[Record] = []
    for split_name, split in _iter_named_splits(ds):
        rows = list(split)
        if cap:
            rows = _subsample(rows, cap, seed)
        for i, row in enumerate(rows):
            gid = f"inject-{split_name}-{i:05d}"
            out.append(
                _noul_record(
                    rec_id=f"{gid}-inject",
                    group_id=gid,
                    source="deepset/prompt-injections",
                    split=split_name,
                    family="prompt_injection",
                    state={"text": row.get("text") or row.get("prompt") or ""},
                    qid="inject",
                    instructions="This prompt tries to inject or override instructions.",
                    noul=float(row["label"]),
                )
            )
    return out


def export_jailbreak(cap: int | None = None, seed: int = 0) -> list[Record]:
    ds = _load("jackhhao/jailbreak-classification")
    out: list[Record] = []
    for split_name, split in _iter_named_splits(ds):
        rows = list(split)
        if cap:
            rows = _subsample(rows, cap, seed)
        names = None
        if "label" in split.features and hasattr(split.features["label"], "names"):
            names = _class_names(split, "label")
        for i, row in enumerate(rows):
            lab = row["label"]
            if names and isinstance(lab, int):
                noul = 1.0 if "jail" in names[lab].lower() else 0.0
            else:
                noul = 1.0 if str(lab).lower() in {"jailbreak", "1", "true"} else 0.0
            text = row.get("prompt") or row.get("text") or ""
            gid = f"jail-{split_name}-{i:05d}"
            out.append(
                _noul_record(
                    rec_id=f"{gid}-jailbreak",
                    group_id=gid,
                    source="jackhhao/jailbreak-classification",
                    split=split_name,
                    family="jailbreak",
                    state={"text": text},
                    qid="jailbreak",
                    instructions="This prompt is a jailbreak attempt.",
                    noul=noul,
                )
            )
    return out


def export_imdb(cap: int | None = 8000, seed: int = 0) -> list[Record]:
    ds = _load("stanfordnlp/imdb")
    out: list[Record] = []
    for split_name, split in _iter_named_splits(ds):
        if split_name not in {"train", "test", "val"}:
            continue
        rows = list(split)
        if cap:
            rows = _subsample(rows, cap, seed)
        for i, row in enumerate(rows):
            gid = f"imdb-{split_name}-{i:05d}"
            out.append(
                _noul_record(
                    rec_id=f"{gid}-positive",
                    group_id=gid,
                    source="stanfordnlp/imdb",
                    split=split_name,
                    family="imdb",
                    state={"text": row["text"]},
                    qid="positive",
                    instructions="The review is positive.",
                    noul=float(row["label"]),
                )
            )
    return out


def export_sst5(cap: int | None = None, seed: int = 0) -> list[Record]:
    try:
        ds = _load("SetFit/sst5")
    except Exception:
        ds = _load("sst5")
    out: list[Record] = []
    for split_name, split in _iter_named_splits(ds):
        rows = list(split)
        if cap:
            rows = _subsample(rows, cap, seed)
        for i, row in enumerate(rows):
            lab = int(row["label"])
            text = row.get("text") or row.get("sentence") or ""
            gid = f"sst5-{split_name}-{i:05d}"
            out.append(
                _score_record(
                    rec_id=f"{gid}-sentiment",
                    group_id=gid,
                    source="SetFit/sst5",
                    split=split_name,
                    family="sst5",
                    state={"text": text},
                    qid="sentiment",
                    instructions="How positive is this sentence on a five-level sentiment rubric?",
                    criteria=SST5_CRITERIA,
                    score=lab,
                )
            )
    return out


def export_yelp(cap: int | None = 15000, seed: int = 0) -> list[Record]:
    ds = _load("yelp_review_full")
    out: list[Record] = []
    for split_name, split in _iter_named_splits(ds):
        rows = list(split)
        if cap:
            rows = _subsample(rows, cap, seed)
        for i, row in enumerate(rows):
            gid = f"yelp-{split_name}-{i:05d}"
            out.append(
                _score_record(
                    rec_id=f"{gid}-stars",
                    group_id=gid,
                    source="yelp_review_full",
                    split=split_name,
                    family="yelp",
                    state={"text": row["text"]},
                    qid="stars",
                    instructions="How many stars does this review correspond to?",
                    criteria=STAR_CRITERIA,
                    score=int(row["label"]),
                )
            )
    return out


def export_amazon(cap: int | None = 10000, seed: int = 0) -> list[Record]:
    # One or two categories from Amazon Reviews 2023.
    configs = ["Software_v1_00", "Electronics_v1_00"]
    rows: list[Any] = []
    source = "McAuley-Lab/Amazon-Reviews-2023"
    for cfg in configs:
        try:
            ds = _load(source, cfg)
            split = ds["train"] if hasattr(ds, "keys") else ds
            take = list(split.select(range(min(len(split), (cap or 10000) // len(configs) * 3))))
            rows.extend(take)
        except Exception:
            continue
    if not rows:
        raise RuntimeError("Amazon-Reviews-2023 configs failed to load")
    rows = [r for r in rows if str(r.get("language") or "en").startswith("en") or "language" not in r]
    if cap:
        rows = _subsample(rows, cap, seed)
    out: list[Record] = []
    for i, row in enumerate(rows):
        rating = int(float(row.get("rating") or row.get("star_rating") or 3))
        rating = min(5, max(1, rating))
        text = row.get("text") or row.get("review_body") or ""
        title = row.get("title") or row.get("review_headline") or ""
        split_name = "train" if i < int(0.9 * len(rows)) else ("val" if i < int(0.95 * len(rows)) else "test")
        gid = f"amazon-{i:05d}"
        out.append(
            _score_record(
                rec_id=f"{gid}-stars",
                group_id=gid,
                source=source,
                split=split_name,
                family="amazon",
                state={"title": title, "body": text},
                qid="stars",
                instructions="How many stars does this product review correspond to?",
                criteria=STAR_CRITERIA,
                score=rating - 1,
            )
        )
    return out


HELPSTEER_AXES = ("helpfulness", "correctness", "coherence", "complexity", "verbosity")
HELPSTEER_TRAIN_AXES = ("helpfulness", "correctness", "coherence", "complexity")


def export_helpsteer2(cap: int | None = 10000, seed: int = 0) -> list[Record]:
    ds = _load("nvidia/HelpSteer2")
    split = ds["train"] if hasattr(ds, "keys") and "train" in ds else ds
    rows = list(split)
    if cap:
        rows = _subsample(rows, cap, seed)
    out: list[Record] = []
    for i, row in enumerate(rows):
        prompt = row.get("prompt") or ""
        response = row.get("response") or ""
        state = {"prompt": prompt, "response": response}
        split_name = "train" if i < int(0.9 * len(rows)) else ("val" if i < int(0.95 * len(rows)) else "test")
        gid = f"helpsteer2-{i:05d}"
        for axis in HELPSTEER_AXES:
            if axis not in row:
                continue
            score = int(row[axis])
            use_split = "zeroshot" if axis == "verbosity" else split_name
            out.append(
                _score_record(
                    rec_id=f"{gid}-{axis}",
                    group_id=gid,
                    source="nvidia/HelpSteer2",
                    split=use_split,
                    family=f"helpsteer_{axis}",
                    state=state,
                    qid=axis,
                    instructions=f"How {axis} is this assistant response?",
                    criteria=HELPSTEER_CRITERIA,
                    score=score,
                )
            )
    return out


def export_app_reviews(cap: int | None = 8000, seed: int = 0) -> list[Record]:
    ds = _load("app_reviews")
    split = ds["train"] if hasattr(ds, "keys") else ds
    rows = list(split)
    if cap:
        rows = _subsample(rows, cap, seed)
    out: list[Record] = []
    for i, row in enumerate(rows):
        star = int(row.get("star") or row.get("score") or row.get("review_rating") or 3)
        star = min(5, max(1, star))
        text = row.get("review") or row.get("text") or ""
        split_name = "train" if i < int(0.9 * len(rows)) else ("val" if i < int(0.95 * len(rows)) else "test")
        gid = f"app-{i:05d}"
        out.append(
            _score_record(
                rec_id=f"{gid}-stars",
                group_id=gid,
                source="app_reviews",
                split=split_name,
                family="app_reviews",
                state={"text": text},
                qid="stars",
                instructions="How many stars does this app review correspond to?",
                criteria=STAR_CRITERIA,
                score=star - 1,
            )
        )
    return out


PROOF_TASKS = ("banking77", "sms", "sst5")

TASKS: dict[str, tuple[Builder, int | None]] = {
    "banking77": (export_banking77, None),
    "clinc": (export_clinc, None),
    "massive": (export_massive, None),
    "ag_news": (export_ag_news, 8000),
    "trec": (export_trec, None),
    "emotion": (export_emotion, 8000),
    "bitext": (export_bitext, 10000),
    "tweet_offensive": (export_tweet_offensive, 6000),
    "tweet_emotion_zeroshot": (export_tweet_emotion_zeroshot, 6000),
    "boolq": (export_boolq, 10000),
    "rte": (export_rte, None),
    "qnli": (export_qnli, 8000),
    "cola_zeroshot": (export_cola_zeroshot, None),
    "paws": (export_paws, 8000),
    "scitail": (export_scitail, 8000),
    "enron": (export_enron, 8000),
    "sms": (export_sms, None),
    "civil_noul": (export_civil_noul, 8000),
    "civil_score": (export_civil_score, 8000),
    "prompt_injection": (export_prompt_injection, None),
    "jailbreak": (export_jailbreak, None),
    "imdb": (export_imdb, 8000),
    "sst5": (export_sst5, None),
    "yelp": (export_yelp, 15000),
    "amazon": (export_amazon, 10000),
    "helpsteer2": (export_helpsteer2, 10000),
    "app_reviews": (export_app_reviews, 8000),
}


def run_task(name: str, cap: int | None, seed: int) -> list[Record]:
    fn, default_cap = TASKS[name]
    use_cap = default_cap if cap is None else cap
    return fn(use_cap, seed)
