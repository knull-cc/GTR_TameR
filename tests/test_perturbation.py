import unittest

import numpy as np
import torch

from utils.perturbation import apply_input_perturbation, perturbation_tag


class ApplyInputPerturbationTest(unittest.TestCase):
    def setUp(self):
        self.batch = torch.tensor(
            [
                [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
                [[-2.0, 5.0], [0.0, 5.0], [2.0, 5.0], [4.0, 5.0]],
            ]
        )

    def test_last_matches_tamer_formula_without_mutating_input(self):
        original = self.batch.clone()
        seed = 2024
        actual = apply_input_perturbation(
            self.batch,
            perturb_type="last",
            perturb_ratio=3.0,
            rng=np.random.RandomState(seed),
        )

        expected = original.clone()
        scale = original.std(dim=1, keepdim=True, unbiased=False)
        noise = torch.as_tensor(
            np.random.RandomState(seed).standard_normal((2, 1, 2)),
            dtype=original.dtype,
        )
        expected[:, -1:, :] += noise * scale * 3.0

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(self.batch, original)
        torch.testing.assert_close(actual[:, :-1, :], original[:, :-1, :])
        # The second channel of the second sample is constant, so its scale is zero.
        self.assertEqual(actual[1, -1, 1].item(), original[1, -1, 1].item())

    def test_none_returns_an_independent_clean_copy(self):
        actual = apply_input_perturbation(
            self.batch,
            perturb_type="none",
            perturb_ratio=3.0,
        )
        torch.testing.assert_close(actual, self.batch)
        self.assertNotEqual(actual.data_ptr(), self.batch.data_ptr())

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "batch_x must have shape"):
            apply_input_perturbation(
                self.batch[0], "last", 3.0, np.random.RandomState(0)
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            apply_input_perturbation(
                self.batch, "last", -1.0, np.random.RandomState(0)
            )
        with self.assertRaisesRegex(ValueError, "rng is required"):
            apply_input_perturbation(self.batch, "last", 3.0)

    def test_result_tag_is_stable(self):
        self.assertEqual(
            perturbation_tag("last", 3.0, 2024),
            "perturb_last_ratio3_seed2024",
        )


if __name__ == "__main__":
    unittest.main()
