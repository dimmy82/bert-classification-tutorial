from src import utils


def convert_annotated_data_jsonl(
    annotated_data_path, labels_dict, features_dict, output_path
):
    """Convert annotated data to validation set format."""
    dev_annotated_data = utils.load_jsonl(annotated_data_path).to_dict(orient="records")
    val_data = []
    for item in dev_annotated_data:
        label = labels_dict[item["label_id"]]
        feature = features_dict[item["feature_id"]]
        val_data.append({"text": feature, "label": label})

    utils.save_jsonl(val_data, output_path)


if __name__ == "__main__":
    dataset_path = "/home/ds/bert-classification-tutorial/datasets/sale-talk-department_generated_company_issue"
    labels_dict = utils.load_json(f"{dataset_path}/label2id.json")
    features_df = utils.load_jsonl(f"{dataset_path}/zero-shot-2025q1/features.jsonl")
    features_dict = dict(zip(features_df["id"], features_df["text"]))

    convert_annotated_data_jsonl(
        f"{dataset_path}/zero-shot-2025q1/train_annotated_data.jsonl",
        labels_dict,
        features_dict,
        f"{dataset_path}/train.jsonl",
    )

    convert_annotated_data_jsonl(
        f"{dataset_path}/zero-shot-2025q1/dev_annotated_data.jsonl",
        labels_dict,
        features_dict,
        f"{dataset_path}/val.jsonl",
    )

    convert_annotated_data_jsonl(
        f"{dataset_path}/zero-shot-2025q1/test_annotated_data.jsonl",
        labels_dict,
        features_dict,
        f"{dataset_path}/test.jsonl",
    )
