from schema import Label, Question, Record
from pack import pack_record, collate_packed
from tests.dummy_tok import DummyTokenizer


def _choice(k=3, state="hello world", choice="a0"):
    criteria = {f"a{i}": f"option {i} description" for i in range(k)}
    return Record(
        id="r",
        group_id="g",
        source="s",
        split="train",
        state={"text": state},
        question=Question(id="q", type="choice", instructions="Pick one.", criteria=criteria),
        label=Label(kind="hard", choice=choice),
    )


def test_mask_count_equals_k():
    tok = DummyTokenizer()
    packed = pack_record(tok, _choice(5), max_length=128)
    assert packed.k == 5
    assert packed.input_ids[0] == tok.cls_token_id
    assert packed.input_ids[-1] == tok.sep_token_id
    for pos in packed.option_positions:
        assert packed.input_ids[pos] == tok.mask_token_id


def test_state_truncated_not_markers():
    tok = DummyTokenizer()
    packed = pack_record(tok, _choice(4, state="word " * 400), max_length=64)
    assert packed.k == 4
    assert len(packed.input_ids) <= 64
    assert len(packed.option_positions) == 4


def test_77_way_fits():
    tok = DummyTokenizer()
    packed = pack_record(tok, _choice(77, state="ticket " * 200, choice="a3"), max_length=512)
    assert packed.k == 77
    assert len(packed.input_ids) <= 512
    assert packed.target_index == packed.option_ids.index("a3")


def test_shuffle_keeps_label_on_option_not_position():
    tok = DummyTokenizer()
    rec = _choice(6, choice="a4")
    a = pack_record(tok, rec, shuffle_options=True, rng=__import__("random").Random(0))
    b = pack_record(tok, rec, shuffle_options=True, rng=__import__("random").Random(1))
    assert a.option_ids[a.target_index] == "a4"
    assert b.option_ids[b.target_index] == "a4"


def test_score_not_shuffled():
    rec = Record(
        id="s",
        group_id="s",
        source="s",
        split="train",
        state="meh",
        question=Question(
            id="stars",
            type="score",
            instructions="stars",
            criteria=["one", "two", "three", "four", "five"],
        ),
        label=Label(kind="hard", score=3),
    )
    tok = DummyTokenizer()
    packed = pack_record(tok, rec, shuffle_options=True, rng=__import__("random").Random(3))
    assert packed.option_ids == ["0", "1", "2", "3", "4"]
    assert packed.target_index == 3


def test_collate_pads_option_positions():
    tok = DummyTokenizer()
    ex = [pack_record(tok, _choice(2)), pack_record(tok, _choice(5, choice="a1"))]
    batch = collate_packed(ex, pad_id=0)
    assert batch["option_positions"].shape[1] == 5
    assert int(batch["option_mask"][0].sum()) == 2
    assert int(batch["option_mask"][1].sum()) == 5
