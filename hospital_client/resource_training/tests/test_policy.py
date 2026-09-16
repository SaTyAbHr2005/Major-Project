from hospital_client.resource_training.policy import ResourcePolicy, default_policy


def test_default_policy_has_sane_bounds():
    p = default_policy()
    assert p.epoch_min <= p.epoch_default <= p.epoch_max
    assert 0.0 < p.vram_safety_margin <= 1.0
    assert 0.0 < p.ram_safety_margin <= 1.0
    assert 1 in p.batch_size_candidates
    assert p.max_adaptation_attempts >= 1
    assert 0.0 < p.batch_size_reduction_factor < 1.0


def test_to_dict_from_dict_round_trip():
    p = default_policy()
    data = p.to_dict()
    restored = ResourcePolicy.from_dict(data)
    assert restored.to_dict() == data


def test_from_dict_allows_partial_override():
    restored = ResourcePolicy.from_dict({"epoch_max": 100})
    assert restored.epoch_max == 100
    assert restored.epoch_min == default_policy().epoch_min


def test_policy_is_mutable_and_independent_per_instance():
    p1 = default_policy()
    p2 = default_policy()
    p1.batch_size_candidates.append(128)
    assert 128 not in p2.batch_size_candidates
