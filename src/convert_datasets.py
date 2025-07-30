from src import utils


def convert_annotated_data_jsonl(
    annotated_data_path, labels_dict, features_dict, output_path
):
    """Convert annotated data to validation set format."""
    # print(f"labels_dict: {labels_dict}.")
    # print(f"features_dict: {features_dict}.")
    annotated_data_set = utils.load_jsonl(annotated_data_path).to_dict(orient="records")
    # print(f"annotated_data_set: {annotated_data_set}.")
    # print("====================")
    val_data = []
    for item in annotated_data_set:
        try:
            label = labels_dict[f"{item['label_id']}"]
            feature = features_dict[f"{item['feature_id']}"]
            val_data.append({"text": feature, "label": label})
        except KeyError as e:
            print(f"KeyError: {e} in item: {item}.")
    # print(f"val_data: {val_data}.")

    utils.save_jsonl(val_data, output_path)


if __name__ == "__main__":
    dataset_path = "/home/ds/bert-classification-tutorial/datasets/sale-talk-department_generated_company_issue"
    labels_dict = utils.load_json(f"{dataset_path}/label2id.json")
    features_df = utils.load_jsonl(
        f"{dataset_path}/zero-shot-2025q1/features.jsonl"
    ).to_dict(orient="records")
    features_dict = {}
    feature_text_set = set()
    for item in features_df:
        feature_text = item["text"]
        if feature_text not in feature_text_set:
            feature_text_set.add(feature_text)
            features_dict[f"{item['id']}"] = feature_text
    # print(f"features_dict: {features_dict.__len__()}.")

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
