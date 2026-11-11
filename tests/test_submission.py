import zipfile

import pandas as pd
import pytest

from st3.submission import pack, read_predictions, validate_archive

HEADER = "painting,language,p_positive\n"
CSV = HEADER + "p1,A,0\np2,B,1\n"


@pytest.fixture
def dev():
    return pd.DataFrame({"painting": ["p1", "p2"], "language": ["A", "B"]})


def test_submission_round_trip(tmp_path, dev):
    csv = tmp_path / "predictions.csv"
    csv.write_text(CSV, encoding="utf-8")
    archive = tmp_path / "submission.zip"
    pack(csv, archive, dev)
    with zipfile.ZipFile(archive) as handle:
        assert handle.namelist() == ["predictions.csv"]
    pd.testing.assert_frame_equal(validate_archive(archive, dev), read_predictions(csv, dev))


@pytest.mark.parametrize("payload", [
    "painting,p_positive,language\np1,.5,A\np2,.5,B\n",
    HEADER + "p1,A,0\n",
    CSV + "p3,C,0\n",
    CSV + "p1,A,0\n",
    HEADER + "p1,A,0,extra\np2,B,1\n",
    HEADER + "p1,A,\np2,B,1\n",
    HEADER + "p1,A,NaN\np2,B,1\n",
    HEADER + "p1,A,inf\np2,B,1\n",
    HEADER + "p1,A,-0.1\np2,B,1\n",
    HEADER + "p1,A,0\np2,B,1.1\n",
])
def test_invalid_prediction_csv(tmp_path, dev, payload):
    path = tmp_path / "predictions.csv"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        read_predictions(path, dev)


def test_misnamed_csv(tmp_path, dev):
    path = tmp_path / "forecast.csv"
    path.write_text(CSV, encoding="utf-8")
    with pytest.raises(ValueError, match="filename"):
        read_predictions(path, dev)


@pytest.mark.parametrize("members", [[], ["nested/predictions.csv"], ["predictions.csv", "notes.txt"]])
def test_invalid_zip_structure(tmp_path, dev, members):
    path = tmp_path / "submission.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name in members:
            archive.writestr(name, CSV)
    with pytest.raises(ValueError, match="top-level"):
        validate_archive(path, dev)
