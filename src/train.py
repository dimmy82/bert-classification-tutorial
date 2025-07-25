from datetime import datetime
from pathlib import Path

from evaluate2.domain.annotated_data import AnnotatedData, AnnotatedDataSet
from evaluate2.domain.evaluated_report import EvaluateSummaryType
from evaluate2.domain.evaluators import MultiLabelEvaluator
from evaluate2.domain.predicted_result import PredictedResult, PredictedResults
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tap import Tap
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer
from evaluate2.domain.label import Labels, Label

import src.utils as utils


class Args(Tap):
    model_name: str = "cl-tohoku/bert-base-japanese-v3"
    dataset_dir: Path = "./datasets/sale-talk-department_generated_company_issue"

    batch_size: int = 4
    epochs: int = 20
    lr: float = 3e-5
    num_warmup_epochs: int = 2
    max_seq_len: int = 1024
    weight_decay: float = 0.01
    gradient_checkpointing: bool = False

    device: str = "cuda:0"
    seed: int = 42

    def process_args(self):
        self.label2id: dict[str, int] = utils.load_json(
            self.dataset_dir / "label2id.json"
        )
        self.labels: list[int] = list(self.label2id.values())
        self.domain_labels = Labels(
            list=[Label(id=f"{label}") for label in self.labels]
        )
        features_df = utils.load_jsonl(
            f"{self.dataset_dir}/zero-shot-2025q1/features.jsonl"
        )
        self.domain_features_text2id = dict(zip(features_df["text"], features_df["id"]))

        date, time = datetime.now().strftime("%Y-%m-%d/%H-%M-%S.%f").split("/")
        self.output_dir = self.make_output_dir(
            "outputs",
            self.model_name,
            date,
            time,
        )

    def make_output_dir(self, *args) -> Path:
        args = [str(a).replace("/", "__") for a in args]
        output_dir = Path(*args)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir


class Experiment:
    def __init__(self, args: Args):
        self.args: Args = args

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            model_max_length=args.max_seq_len,
        )

        self.model: PreTrainedModel = (
            AutoModelForSequenceClassification.from_pretrained(
                args.model_name,
                num_labels=len(args.labels),
            )
            .eval()
            .to(args.device, non_blocking=True)
        )

        # gradient_checkpointingとtorch.compileは相性が悪いことが多いので排他的に使用
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        else:
            self.model = torch.compile(self.model)

        self.train_dataloader: DataLoader = self.load_dataset(
            split="train", shuffle=True
        )
        self.val_dataloader: DataLoader = self.load_dataset(split="val")
        self.test_dataloader: DataLoader = self.load_dataset(split="test")

        self.optimizer, self.lr_scheduler = self.create_optimizer()

    def load_dataset(
        self,
        split: str,
        shuffle: bool = False,
    ) -> DataLoader:
        path: Path = self.args.dataset_dir / f"{split}.jsonl"
        dataset: list[dict] = utils.load_jsonl(path).to_dict(orient="records")
        return self.create_loader(dataset, shuffle=shuffle)

    def collate_fn(self, data_list: list[dict]) -> dict:
        texts = [d["text"] for d in data_list]
        # title = [d["title"] for d in data_list]
        # body = [d["body"] for d in data_list]

        inputs: BatchEncoding = self.tokenizer(
            texts,
            # title,
            # body,
            padding=True,
            # truncation="only_second",
            return_tensors="pt",
            max_length=args.max_seq_len,
        )

        labels = [d["label"] for d in data_list]
        # print("======== labels ========")
        # print(labels)
        labels_tensor = torch.LongTensor(labels)
        # print(labels_tensor)
        return {
            "feature_ids": [self.args.domain_features_text2id[text] for text in texts],
            "batch": BatchEncoding({**inputs, "labels": labels_tensor}),
        }

    def create_loader(
        self,
        dataset,
        batch_size=None,
        shuffle=False,
    ):
        return DataLoader(
            dataset,
            collate_fn=self.collate_fn,
            batch_size=batch_size or args.batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
        )

    def create_optimizer(
        self,
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        # see: https://tma15.github.io/blog/2021/09/17/deep-learningbert%E5%AD%A6%E7%BF%92%E6%99%82%E3%81%ABbias%E3%82%84layer-normalization%E3%82%92weight-decay%E3%81%97%E3%81%AA%E3%81%84%E7%90%86%E7%94%B1/
        no_decay = {"bias", "LayerNorm.weight"}
        optimizer_grouped_parameters = [
            {
                "params": [
                    param
                    for name, param in self.model.named_parameters()
                    if not name in no_decay
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    param
                    for name, param in self.model.named_parameters()
                    if name in no_decay
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=len(self.train_dataloader) * args.num_warmup_epochs,
            num_training_steps=len(self.train_dataloader) * args.epochs,
        )

        return optimizer, lr_scheduler

    # @torch.cuda.amp.autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else None)
    @torch.cuda.amp.autocast(enabled=True, dtype=torch.float16)
    def run(self):
        best_score_key = "recall"
        start_metrics = self.evaluate2(self.test_dataloader)
        self.log(start_metrics)
        val_metrics = self.evaluate2(self.val_dataloader)
        best_epoch, best_score = None, val_metrics[best_score_key]
        best_state_dict = self.clone_state_dict()
        print("====================== org_model ======================")
        print(self.model)
        # print("====================== org_state_dict ======================")
        # print(best_state_dict)

        scaler = torch.cuda.amp.GradScaler()

        for epoch in trange(args.epochs, dynamic_ncols=True):
            self.model.train()

            for batch_dict in tqdm(
                self.train_dataloader,
                total=len(self.train_dataloader),
                dynamic_ncols=True,
                leave=False,
            ):
                batch = batch_dict["batch"]
                out: SequenceClassifierOutput = self.model(**batch.to(args.device))
                loss: torch.FloatTensor = out.loss

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)

                scale = scaler.get_scale()
                scaler.update()
                if scale <= scaler.get_scale():
                    self.lr_scheduler.step()

            self.model.eval()
            val_metrics = {"epoch": epoch, **self.evaluate2(self.val_dataloader)}

            # 開発セットでのF値最良時のモデルを保存
            if val_metrics[best_score_key] > best_score:
                best_epoch = epoch
                best_score = val_metrics[best_score_key]
                best_state_dict = self.clone_state_dict()
                self.log(val_metrics)
                # print("====================== better_model ======================")
                # print(self.model)
                # print("====================== best_state_dict ======================")
                # print(best_state_dict)

        self.model.load_state_dict(best_state_dict)
        self.model.eval().to(args.device, non_blocking=True)

        val_metrics = {"best-epoch": best_epoch, **self.evaluate2(self.val_dataloader)}
        test_metrics = self.evaluate2(self.test_dataloader)

        return start_metrics, val_metrics, test_metrics

    @torch.no_grad()
    # @torch.cuda.amp.autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else None)
    @torch.cuda.amp.autocast(enabled=True, dtype=torch.float16)
    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss, gold_labels, pred_labels = 0, [], []

        for batch_dict in tqdm(
            dataloader, total=len(dataloader), dynamic_ncols=True, leave=False
        ):
            batch = batch_dict["batch"]
            out: SequenceClassifierOutput = self.model(**batch.to(self.args.device))

            batch_size: int = batch.input_ids.size(0)
            loss = out.loss.item() * batch_size
            total_loss += loss

            pred_labels += out.logits.argmax(dim=-1).tolist()
            gold_labels += batch.labels.tolist()

        accuracy: float = accuracy_score(gold_labels, pred_labels)
        # macro top 1 の評価
        precision, recall, f1, _ = precision_recall_fscore_support(
            gold_labels,
            pred_labels,
            average="macro",
            zero_division=0,
            labels=args.labels,
        )

        return {
            "loss": loss / len(dataloader.dataset),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    @torch.no_grad()
    # @torch.cuda.amp.autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else None)
    @torch.cuda.amp.autocast(enabled=True, dtype=torch.float16)
    def evaluate2(self, dataloader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss, feature_ids, true_labels, predicted_labels = 0, [], [], []

        for batch_dict in tqdm(
            dataloader, total=len(dataloader), dynamic_ncols=True, leave=False
        ):
            feature_ids += batch_dict["feature_ids"]
            batch = batch_dict["batch"]
            # print(f"***: {feature_ids} == {batch['labels']}")
            out: SequenceClassifierOutput = self.model(**batch.to(self.args.device))

            batch_size: int = batch.input_ids.size(0)
            loss = out.loss.item() * batch_size
            total_loss += loss

            true_labels += batch.labels.tolist()
            predicted_labels += out.logits.argmax(dim=-1).tolist()

        # feature_idsとtrue_labelsを融合した配列を作りたい
        annotated_data_set = AnnotatedDataSet(
            set={
                AnnotatedData(feature_id=str(feature_id), label_id=str(label_id))
                for feature_id, label_id in zip(feature_ids, true_labels)
            }
        )
        predicted_results = PredictedResults(
            list=[
                PredictedResult(
                    feature_id=str(feature_id),
                    label_id=str(label_id),
                    predicted_similarity=1.0,
                )
                for feature_id, label_id in zip(feature_ids, predicted_labels)
            ]
        )

        evaluator = MultiLabelEvaluator.load(
            self.args.domain_labels, annotated_data_set, predicted_results
        )

        return {
            "loss": loss / len(dataloader.dataset),
            "accuracy": evaluator.pr_auc(EvaluateSummaryType.MICRO.value),
            "precision": evaluator.precision(1, EvaluateSummaryType.MICRO.value),
            "recall": evaluator.recall(1, EvaluateSummaryType.MICRO.value),
            "f1": 0.0,
        }

    def log(self, metrics: dict) -> None:
        utils.log(metrics, self.args.output_dir / "log.csv")
        tqdm.write(
            f"epoch: {metrics.get('epoch', None)} \t"
            f"loss: {metrics['loss']:2.6f}   \t"
            f"accuracy: {metrics['accuracy']:.4f} \t"
            f"precision: {metrics['precision']:.4f} \t"
            f"recall: {metrics['recall']:.4f} \t"
            f"f1: {metrics['f1']:.4f}"
        )

    def clone_state_dict(self) -> dict:
        return {k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()}


def main(args: Args):
    exp = Experiment(args=args)
    start_metrics, val_metrics, test_metrics = exp.run()

    utils.save_json(start_metrics, args.output_dir / "start-metrics.json")
    utils.save_json(val_metrics, args.output_dir / "val-metrics.json")
    utils.save_json(test_metrics, args.output_dir / "test-metrics.json")
    utils.save_config(args, args.output_dir / "config.json")


if __name__ == "__main__":
    print(f"is_bf16_supported: {torch.cuda.is_bf16_supported()}")
    args = Args().parse_args()
    utils.init(seed=args.seed)
    main(args)
