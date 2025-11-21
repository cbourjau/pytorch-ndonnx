import ndonnx as ndx
import numpy as np
import onnxruntime as ort
import pytest
import torch
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    pipeline,
)

from pytorch_ndonnx import Tensor, monkey_patches


def test_sentiment():
    classifier = pipeline(task="sentiment-analysis")  # type: ignore
    # classifier.model = jit(classifier.model)
    raw_inputs = "Hugging Face is the best thing since sliced bread!"
    processed_inputs = classifier.preprocess(raw_inputs)

    expected_logits = classifier.forward(processed_inputs)["logits"]
    # preds = classifier("Hugging Face is the best thing since sliced bread!")
    # preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]

    # Prepare inputs
    ins = {k: Tensor(v) for k, v in processed_inputs.items()}  # type: ignore
    with monkey_patches():
        candidate_logits = classifier.forward(ins)["logits"]

    np.testing.assert_allclose(
        candidate_logits.unwrap_numpy(), expected_logits.numpy(), rtol=1e-5
    )


def test_distilbert():
    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english", revision="714eb0f"
    )
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english", revision="714eb0f"
    )

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    expected_logits = model(**inputs).logits

    # Test build using dynamic dimensions rather than constants
    ins = {
        "attention_mask": Tensor(ndx.argument(shape=("N", "M"), dtype=ndx.int64)),
        "input_ids": Tensor(ndx.argument(shape=("N", "M"), dtype=ndx.int64)),
    }
    with monkey_patches():
        logits = model(**ins).logits

    model = ndx.build({k: v._inner for k, v in ins.items()}, {"logits": logits._inner})
    sess = ort.InferenceSession(model.SerializeToString())
    (candidate_logits,) = sess.run(None, {k: v.numpy() for k, v in inputs.items()})

    np.testing.assert_allclose(
        candidate_logits, expected_logits.detach().numpy(), rtol=1e-6
    )


@pytest.mark.skip(reason="For debugging only")
def test_upstream_onnx_conversion():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # load model and tokenizer
    model_id = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dummy_model_input = tokenizer("This is a sample", return_tensors="pt")

    # export
    torch.onnx.export(
        model,
        tuple(dummy_model_input.values()),
        f="torch-model.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence"},
            "attention_mask": {0: "batch_size", 1: "sequence"},
            "logits": {0: "batch_size", 1: "sequence"},
        },
        do_constant_folding=True,
        opset_version=21,
        dynamo=True,
    )
