import json

from schema import Label, Question, Record, record_from_dict, serialize_state


def _rec():
    return Record(
        id="banking77-train-00412-intent",
        group_id="banking77-train-00412",
        source="PolyAI/banking77",
        split="train",
        state={"text": "I still have not received my new card"},
        question=Question(
            id="intent",
            type="choice",
            instructions="Which banking intent matches this customer message?",
            criteria={
                "card_arrival": "The user is waiting for a physical card to arrive",
                "activate_my_card": "The user wants to activate a card",
                "other": "Anything else",
            },
        ),
        label=Label(kind="hard", choice="card_arrival"),
    )


def test_roundtrip_matches_plan_example():
    rec = _rec()
    rec.teacher = None
    d = rec.to_dict()
    assert d["id"] == "banking77-train-00412-intent"
    assert d["question"]["type"] == "choice"
    assert d["label"]["choice"] == "card_arrival"
    back = record_from_dict(json.loads(json.dumps(d)))
    assert back.hard_option_id() == "card_arrival"


def test_noul_hard_option():
    rec = Record(
        id="n",
        group_id="n",
        source="s",
        split="train",
        state={"text": "buy cheap viagra"},
        question=Question(id="spam", type="noul", instructions="This message is spam."),
        label=Label(kind="hard", noul=1),
    )
    assert rec.hard_option_id() == "true"
    rec.label.noul = 0
    assert rec.hard_option_id() == "false"


def test_serialize_state_text_and_json():
    assert serialize_state({"text": "hi"}) == "hi"
    s = serialize_state({"from": "a", "body": "b"})
    assert "from" in s and "body" in s
