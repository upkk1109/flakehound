 """M1 rule tests -- the TP/FP-guard pattern every rule must follow."""

  from __future__ import annotations

  import ast

  from flakehound.rules.base import Confidence, FileContext
  from flakehound.rules.m1_unseeded_stochastic_assert import UnseededStochasticAssert


  def _ctx(source: str, name: str = "test_x.py") -> FileContext:
      return FileContext(
          path=name,
          source=source,
          tree=ast.parse(source),
          is_test_file=True,
          is_conftest=False,
      )


  def _run(source: str):
      return list(UnseededStochasticAssert().check(_ctx(source)))


  def test_detects_unseeded_assert_allclose_on_random_normal():
      src = (
          "import numpy as np\n"
          "\n"
          "def test_a():\n"
          "    x = np.random.normal(size=5)\n"
          "    np.testing.assert_allclose(x, [0.1, 0.2, 0.3, 0.4, 0.5])\n"
      )
      findings = _run(src)
      assert len(findings) == 1
      assert findings[0].rule_id == "M1"
      assert findings[0].line == 5
      assert findings[0].confidence == Confidence.MEDIUM


  def test_detects_unseeded_pytest_approx_on_direct_random_call():
      src = (
          "import pytest\n"
          "import numpy as np\n"
          "\n"
          "def test_b():\n"
          "    assert np.random.uniform() == pytest.approx(0.5)\n"
      )
      findings = _run(src)
      assert len(findings) == 1
      assert findings[0].rule_id == "M1"
      assert findings[0].line == 5


  def test_model_fit_predict_is_weaker_evidence_advisory():
      src = (
          "import pytest\n"
          "\n"
          "def test_c(model, X_train, y_train, X_test):\n"
          "    model.fit(X_train, y_train)\n"
          "    pred = model.predict(X_test)\n"
          "    assert pred.mean() == pytest.approx(0.5)\n"
      )
      findings = _run(src)
      assert len(findings) == 1
      assert findings[0].rule_id == "M1"
      assert findings[0].line == 6
      assert findings[0].confidence == Confidence.ADVISORY


  def test_fp_guard_seed_established_in_test_is_clean():
      src = (
          "import numpy as np\n"
          "\n"
          "def test_d():\n"
          "    rng = np.random.default_rng(42)\n"
          "    x = rng.normal(size=5)\n"
          "    np.testing.assert_allclose(x, [0.1, 0.2, 0.3, 0.4, 0.5])\n"
      )
      assert _run(src) == []


  def test_fp_guard_seed_established_in_fixture_dependency_is_clean():
      src = (
          "import numpy as np\n"
          "import pytest\n"
          "\n"
          "@pytest.fixture\n"
          "def rng():\n"
          "    return np.random.default_rng(0)\n"
          "\n"
          "def test_e(rng):\n"
          "    x = rng.normal(size=3)\n"
          "    np.testing.assert_allclose(x, [0.1, 0.2, 0.3])\n"
      )
      assert _run(src) == []


  def test_fp_guard_deterministic_transform_comparison_is_clean():
      src = (
          "import numpy as np\n"
          "\n"
          "def test_f():\n"
          "    x = np.array([1.0, 2.0, 3.0]) * 2\n"
          "    np.testing.assert_allclose(x, [2.0, 4.0, 6.0])\n"
      )
      assert _run(src) == []


  def test_fp_guard_static_fixture_golden_comparison_is_clean():
      src = (
          "import numpy as np\n"
          "\n"
          "def test_g(golden_array):\n"
          "    actual = np.load('fixture.npy')\n"
          "    np.testing.assert_allclose(actual, golden_array)\n"
      )
      assert _run(src) == []


  def test_detects_unseeded_torch_randn():
      src = (
          "import torch\n"
          "import numpy as np\n"
          "\n"
          "def test_h():\n"
          "    x = torch.randn(5)\n"
          "    np.testing.assert_allclose(x, [0.1, 0.2, 0.3, 0.4, 0.5])\n"
      )
      findings = _run(src)
      assert len(findings) == 1
      assert findings[0].rule_id == "M1"
      assert findings[0].line == 6


  def test_detects_unseeded_torch_rand():
      src = (
          "import torch\n"
          "import numpy as np\n"
          "\n"
          "def test_i():\n"
          "    x = torch.rand(3, 3)\n"
          "    np.testing.assert_allclose(x, [[0.1, 0.2, 0.3]] * 3)\n"
      )
      findings = _run(src)
      assert len(findings) == 1
      assert findings[0].rule_id == "M1"


  def test_fp_guard_torch_randn_with_generator_seed_is_clean():
      src = (
          "import torch\n"
          "import numpy as np\n"
          "\n"
          "def test_j():\n"
          "    g = torch.Generator().manual_seed(42)\n"
          "    x = torch.randn(5, generator=g)\n"
          "    np.testing.assert_allclose(x, [0.1, 0.2, 0.3, 0.4, 0.5])\n"
      )
      assert _run(src) == []


  def test_fp_guard_random_seed_establishes_determinism():
      src = (
          "import random\n"
          "import numpy as np\n"
          "\n"
          "def test_k():\n"
          "    random.seed(42)\n"
          "    x = random.random()\n"
          "    np.testing.assert_allclose(x, 0.5)\n"
      )
      assert _run(src) == []


  def test_fp_guard_non_stochastic_rand_name_is_clean():
      """`brand_total` contains 'rand' as a substring but is not a random op."""
      src = (
          "import numpy as np\n"
          "\n"
          "def test_l(brands):\n"
          "    total = brands.total()\n"
          "    np.testing.assert_allclose(total, 100.0)\n"
      )
      assert _run(src) == []
