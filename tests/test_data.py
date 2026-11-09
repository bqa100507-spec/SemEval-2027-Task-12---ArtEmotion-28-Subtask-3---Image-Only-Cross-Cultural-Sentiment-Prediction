import pandas as pd
import pytest

from st3.data import aggregate, check_dev_isolation, check_split_integrity, read_dev_input, validation_gold


def test_sentiment_mapping_and_threshold():
    rows = [("paint", "A", emotion) for emotion in
            ("amusement", "awe", "contentment", "excitement", "fear", "other")]
    rows += [("paint", "B", emotion) for emotion in ("anger", "disgust", "fear", "sadness", "something else")]
    rows += [("paint", "C", emotion) for emotion in ("anger", "disgust", "fear", "sadness", "sadness")]
    result = aggregate(pd.DataFrame(rows, columns=["painting", "language", "emotion"]))
    assert list(result.itertuples(index=False, name=None)) == [("paint", "A", 4, 5, 0.8), ("paint", "C", 0, 5, 0.0)]


@pytest.mark.parametrize(("paintings", "paths", "message"), [
    (["x", "x"], ["x.jpg", "x.jpg"], "both train and val"),
    (["x", "y"], ["x.jpg", "x.jpg"], "one-to-one"),
])
def test_split_cannot_leak_by_painting_or_image(paintings, paths, message):
    frame = pd.DataFrame({"painting": paintings, "image_name": paths, "split": ["train", "val"]})
    with pytest.raises(ValueError, match=message):
        check_split_integrity(frame)


def test_dev_must_use_validation_ids_and_paths():
    frame = pd.DataFrame({"painting": ["x", "y"], "image_name": ["x.jpg", "y.jpg"], "split": ["train", "val"]})
    check_split_integrity(frame)
    dev = pd.DataFrame({"painting": ["y"], "image": ["y.jpg"]})
    check_dev_isolation(frame, dev)
    with pytest.raises(ValueError, match="overlaps training"):
        check_dev_isolation(frame, dev.assign(image="x.jpg"))
    with pytest.raises(ValueError, match="do not match"):
        check_dev_isolation(frame, dev.assign(image="other.jpg"))


def test_dev_csv_rejects_extra_inputs_and_duplicate_keys(tmp_path):
    path = tmp_path / "phase_inputs" / "dev" / "st3_test.csv"
    path.parent.mkdir(parents=True)
    path.write_text("painting,language,image,caption\nx,A,x.jpg,caption\n", encoding="utf-8")
    with pytest.raises(ValueError, match="columns"):
        read_dev_input(tmp_path)
    path.write_text("painting,language,image\nx,A,x.jpg\nx,A,x.jpg\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        read_dev_input(tmp_path)


def test_dev_gold_coverage():
    frame = pd.DataFrame([("val", "A", "awe", "val")] * 5 + [("train", "B", "fear", "train")] * 5,
                         columns=["painting", "language", "emotion", "split"])
    with pytest.raises(ValueError, match="1 missing, 1 extra"):
        validation_gold(frame, pd.DataFrame({"painting": ["train"], "language": ["B"]}))
