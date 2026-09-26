import pytest

from eval_real_indy_rg2 import _get_warmup_observation_with_retry
from umi.real_world.rg2ft_obs import FTObservationStaleError


class _WarmupEnv:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def get_obs(self, *, include_valve_context_stream):
        assert include_valve_context_stream is False
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_warmup_retries_only_stale_ft_until_fresh_observation():
    expected = {"timestamp": [1.0]}
    env = _WarmupEnv(
        [
            FTObservationStaleError("dual-F/T observation is stale"),
            FTObservationStaleError("dual-F/T observation is stale"),
            expected,
        ]
    )
    sleeps = []

    actual = _get_warmup_observation_with_retry(
        env,
        timeout_s=3.0,
        retry_interval_s=0.02,
        monotonic_func=lambda: 0.0,
        sleep_func=sleeps.append,
    )

    assert actual is expected
    assert env.calls == 3
    assert sleeps == [0.02, 0.02]


def test_warmup_stale_ft_still_fails_after_bounded_timeout():
    env = _WarmupEnv(
        [FTObservationStaleError("dual-F/T observation is stale")]
    )
    times = iter([10.0, 13.0])

    with pytest.raises(FTObservationStaleError, match="within 3.00s after 1 retries"):
        _get_warmup_observation_with_retry(
            env,
            timeout_s=3.0,
            retry_interval_s=0.02,
            monotonic_func=lambda: next(times),
            sleep_func=lambda _: None,
        )

    assert env.calls == 1


def test_warmup_does_not_retry_unrelated_observation_errors():
    env = _WarmupEnv([RuntimeError("camera worker failed")])

    with pytest.raises(RuntimeError, match="camera worker failed"):
        _get_warmup_observation_with_retry(
            env,
            monotonic_func=lambda: 0.0,
            sleep_func=lambda _: None,
        )

    assert env.calls == 1
