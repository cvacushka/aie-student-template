from airguard.data.generate_dataset import generate_dataset
from airguard.data.validate import validate_dataset
from airguard.features import TARGET_COLUMN, build_feature_frame


def test_generated_dataset_has_expected_shape_and_target() -> None:
    dataset = generate_dataset(rows=500, seed=7)

    assert len(dataset) == 500
    assert TARGET_COLUMN in dataset.columns
    assert dataset[TARGET_COLUMN].nunique() == 2
    assert validate_dataset(dataset) == []
    assert not build_feature_frame(dataset).isna().any().any()
