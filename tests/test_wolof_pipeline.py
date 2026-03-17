from __future__ import annotations

from pathlib import Path

from galsenai_llm.wolof_pipeline import (
    ConversationRecord,
    GpuProfile,
    _build_markdown_report,
    _resolve_benchmark_methods,
)


def _sample_gpu_profile() -> GpuProfile:
    return GpuProfile(
        detected=True,
        device_name="Test GPU",
        total_memory_mib=8192,
        bf16_supported=True,
        precision="bf16",
        training_strategy="qlora",
        benchmark_batch_size=1,
        benchmark_grad_accum=8,
        benchmark_max_length_cap=512,
        benchmark_eval_batch_size=1,
        benchmark_packing=False,
        full_batch_size=1,
        full_grad_accum=16,
        full_max_length_cap=768,
        full_eval_batch_size=1,
        full_packing=False,
        quantization_bits=4,
        nvidia_smi_output="Test GPU, 8192",
    )


def test_resolve_benchmark_methods_defaults() -> None:
    assert _resolve_benchmark_methods(include_method_c=False, force_method=None) == [
        "method_a",
        "method_b",
    ]
    assert _resolve_benchmark_methods(include_method_c=True, force_method=None) == [
        "method_a",
        "method_b",
        "method_c",
    ]


def test_resolve_benchmark_methods_force_method() -> None:
    assert _resolve_benchmark_methods(include_method_c=False, force_method="method_a") == [
        "method_a",
    ]
    assert _resolve_benchmark_methods(include_method_c=False, force_method="method_c") == [
        "method_c",
    ]


def test_build_markdown_report_mentions_forced_method() -> None:
    report = _build_markdown_report(
        output_dir=Path("outputs/test-run"),
        gpu_profile=_sample_gpu_profile(),
        model_name="Qwen/Qwen2.5-3B-Instruct",
        method_a_text_corpus_file=Path("data/method_a.txt"),
        method_a_text_count=10,
        benchmark_results=[
            {
                "method": "method_a",
                "fertility": 1.5,
                "coverage": 48.0,
                "embedding_init_quality": 0.62,
                "perplexity": 76.5,
            }
        ],
        winner={"method": "method_a"},
        final_result={
            "sample_generations": [],
            "method": "method_a",
            "eval_loss": 4.5,
            "perplexity": 90.7,
            "training_plan": {"num_train_epochs": 2},
            "parameter_counts": {"total": 1, "trainable": 1},
        },
        benchmark_splits={
            "train": [ConversationRecord("a", [], 0, 0)],
            "val": [],
            "test": [],
        },
        final_splits={
            "train": [ConversationRecord("b", [], 0, 0)],
            "val": [],
        },
        english_replay_metadata=None,
        math_replay_metadata=None,
        train_mix_metadata=None,
        final_train_record_count=1,
        forced_method="method_a",
    )

    assert "Forced tokenizer method: `method_a`" in report
    assert "automatic tokenizer method selection was skipped" in report
