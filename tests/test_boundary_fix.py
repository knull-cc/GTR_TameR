import unittest

import torch

from utils.boundary_fix import apply_boundary_fix


class ApplyBoundaryFixTest(unittest.TestCase):
    def test_latest_point_is_replaced_without_mutating_input(self):
        batch = torch.tensor(
            [
                [[1.0, 10.0], [2.0, 20.0], [9.0, 90.0]],
                [[-1.0, 5.0], [3.0, 6.0], [8.0, 7.0]],
            ]
        )
        original = batch.clone()

        fixed = apply_boundary_fix(batch)

        torch.testing.assert_close(fixed[:, :-1, :], original[:, :-1, :])
        torch.testing.assert_close(fixed[:, -1, :], original[:, -2, :])
        torch.testing.assert_close(batch, original)
        self.assertNotEqual(fixed.data_ptr(), batch.data_ptr())

    def test_invalid_input_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            apply_boundary_fix(torch.ones(4, 2))

    def test_sequence_must_have_two_points(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            apply_boundary_fix(torch.ones(2, 1, 3))


if __name__ == "__main__":
    unittest.main()
