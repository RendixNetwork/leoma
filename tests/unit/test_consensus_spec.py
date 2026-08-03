"""The consensus surface: no defaults, no extras, no silent drift.

Every test here pins a way two honest validators could have reached different
verdicts on the same challenger — which is the failure mode that matters most,
because it is silent, plausible, and destroys the subnet's credibility without
anyone noticing.
"""

import pytest
from pydantic import ValidationError

from leoma.eval.digests import canonical_json, digest_obj
from leoma.eval.errors import ConsensusConfigError
from leoma.eval.spec import (
    ArchSpec,
    ConsensusSpec,
    CorpusSpec,
    DuelSpec,
    EvalRuntimeSpec,
    GenSpec,
    verify_echo,
)

from .conftest import pinned_spec


class TestNoDefaults:
    """A field with a default is a field a validator can silently forget."""

    @pytest.mark.parametrize("missing", ["metric", "metric_device", "n_clips",
                                         "delta_threshold", "alpha", "n_bootstrap",
                                         "early_stop_enabled"])
    def test_a_missing_duel_field_raises(self, missing):
        spec = pinned_spec()
        fields = spec.duel.model_dump()
        fields.pop(missing)
        with pytest.raises(ValidationError):
            DuelSpec(**fields)

    @pytest.mark.parametrize("missing", ["num_frames", "fps", "width", "height",
                                         "guidance_scale", "num_inference_steps",
                                         "negative_prompt", "prompt_mode", "prompt", "dtype"])
    def test_a_missing_gen_field_raises(self, missing):
        spec = pinned_spec()
        fields = spec.gen.model_dump()
        fields.pop(missing)
        with pytest.raises(ValidationError):
            GenSpec(**fields)

    def test_an_unknown_field_raises(self):
        spec = pinned_spec()
        with pytest.raises(ValidationError):
            DuelSpec(**spec.duel.model_dump(), future_knob=1)

    @pytest.mark.parametrize("missing", ["base_repo", "base_digest", "pipeline"])
    def test_a_missing_arch_field_raises(self, missing):
        spec = pinned_spec()
        fields = spec.arch.model_dump()
        fields.pop(missing)
        with pytest.raises(ValidationError):
            ArchSpec(**fields)

    @pytest.mark.parametrize(
        "missing", ["python", "cuda", "compute_capability", "eval_lock_digest"]
    )
    def test_a_missing_runtime_field_raises(self, missing):
        spec = pinned_spec()
        fields = spec.runtime.model_dump()
        fields.pop(missing)
        with pytest.raises(ValidationError):
            EvalRuntimeSpec(**fields)


class TestDigest:
    def test_same_spec_same_digest(self):
        assert pinned_spec().digest() == pinned_spec().digest()

    def test_any_field_changes_the_digest(self):
        base = pinned_spec()
        moved = base.model_copy(
            update={"duel": base.duel.model_copy(update={"n_clips": base.duel.n_clips + 1})}
        )
        assert moved.digest() != base.digest()

    def test_base_architecture_digest_changes_the_consensus_digest(self):
        base = pinned_spec()
        replacement = "sha256:" + "a" * 64
        moved = base.model_copy(
            update={"arch": base.arch.model_copy(update={"base_digest": replacement})}
        )
        assert moved.digest() != base.digest()

    def test_runtime_lock_changes_the_consensus_digest(self):
        base = pinned_spec()
        moved = base.model_copy(
            update={
                "runtime": base.runtime.model_copy(
                    update={"eval_lock_digest": "sha256:" + "f" * 64}
                )
            }
        )
        assert moved.digest() != base.digest()

    def test_digest_survives_a_json_round_trip(self):
        """The spec crosses HTTP. If a float came back with a different last bit the
        digest would flip and every duel would be refused — so floats are quantized."""
        import json

        base = pinned_spec()
        round_tripped = ConsensusSpec.model_validate(json.loads(json.dumps(base.model_dump(mode="json"))))
        assert round_tripped.digest() == base.digest()

    def test_key_order_does_not_matter(self):
        assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})
        assert digest_obj({"b": 1, "a": 2}) == digest_obj({"a": 2, "b": 1})


class TestVerifyEcho:
    """The check that catches "the field wasn't sent, so the box used its default"."""

    def test_matching_echo_passes(self):
        spec = pinned_spec()
        verify_echo(spec, spec.model_dump(mode="json"))

    def test_no_echo_at_all_is_refused(self):
        # An eval box on a build that predates the consensus surface. It may well
        # have used its own generation parameters — and it wouldn't know to say so.
        with pytest.raises(ConsensusConfigError, match="no consensus echo"):
            verify_echo(pinned_spec(), None)

    def test_a_box_that_ran_a_different_metric_is_caught(self):
        spec = pinned_spec()
        echoed = spec.model_dump(mode="json")
        echoed["duel"]["metric"] = "mse"
        with pytest.raises(ConsensusConfigError, match="DIFFERENT consensus spec"):
            verify_echo(spec, echoed)

    def test_the_error_names_the_field_that_drifted(self):
        spec = pinned_spec()
        echoed = spec.model_dump(mode="json")
        echoed["gen"]["num_frames"] = 17
        with pytest.raises(ConsensusConfigError, match=r"gen\.num_frames: sent=81 echoed=17"):
            verify_echo(spec, echoed)

    def test_a_malformed_echo_is_refused(self):
        with pytest.raises(ConsensusConfigError, match="malformed"):
            verify_echo(pinned_spec(), {"duel": "not a spec"})


class TestCorpusPin:
    def test_the_shipped_chain_toml_is_pinned_and_duel_ready(self):
        """The operator-published manifest is the only corpus validators may use."""
        from leoma.infra.chain_config import SPEC

        assert SPEC.corpus.pinned
        SPEC.require_duel_ready()

    def test_a_pinned_corpus_is_duel_ready(self):
        pinned_spec().require_duel_ready()  # does not raise

    def test_a_garbage_digest_is_rejected_outright(self):
        with pytest.raises(ValidationError):
            CorpusSpec(bucket="videos", manifest_key="m.json", manifest_digest="not-a-hash")


class TestEarlyStopBound:
    def test_shipped_consensus_disables_unproven_bound(self):
        spec = pinned_spec()
        assert spec.duel.early_stop_enabled is False
        assert spec.early_stop_max_advantage is None

    def test_enabled_bound_is_derived_from_pinned_factor(self):
        spec = pinned_spec()
        enabled = spec.model_copy(
            update={"duel": spec.duel.model_copy(update={"early_stop_enabled": True})}
        )
        assert enabled.early_stop_max_advantage == pytest.approx(
            enabled.duel.early_stop_factor * enabled.duel.delta_threshold
        )

    def test_enablement_changes_consensus_digest(self):
        spec = pinned_spec()
        enabled = spec.model_copy(
            update={"duel": spec.duel.model_copy(update={"early_stop_enabled": True})}
        )
        assert enabled.digest() != spec.digest()
