import unittest

import torch

from utils.boundary_reconstruction import (
    BoundaryReconstructor,
    apply_filtered_boundary_reconstruction,
)


class BoundaryReconstructorTest(unittest.TestCase):
    def test_prediction_does_not_depend_on_true_last_point(self):
        torch.manual_seed(7)
        model = BoundaryReconstructor(seq_len=4, channels=2, hidden_dim=8)
        first = torch.tensor(
            [[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]]
        )
        second = first.clone()
        second[:, -1, :] = torch.tensor([[400.0, -500.0]])

        torch.testing.assert_close(model(first), model(second))

    def test_only_extreme_boundary_channels_are_replaced(self):
        batch = torch.tensor(
            [
                [
                    [0.0, 0.0],
                    [1.0, 1.0],
                    [2.0, 2.0],
                    [3.0, 3.0],
                    [20.0, 4.0],
                ]
            ]
        )
        predicted_last = torch.tensor([[4.0, 99.0]])
        original = batch.clone()

        fixed, mask, scores = apply_filtered_boundary_reconstruction(
            batch,
            predicted_last,
            threshold=3.0,
        )

        self.assertEqual(mask.tolist(), [[True, False]])
        self.assertGreater(scores[0, 0].item(), 3.0)
        self.assertLessEqual(scores[0, 1].item(), 3.0)
        self.assertEqual(fixed[0, -1, 0].item(), 4.0)
        self.assertEqual(fixed[0, -1, 1].item(), 4.0)
        torch.testing.assert_close(fixed[:, :-1, :], original[:, :-1, :])
        torch.testing.assert_close(batch, original)

    def test_invalid_shapes_and_threshold_are_rejected(self):
        batch = torch.ones(2, 4, 3)
        predicted = torch.ones(2, 3)

        with self.assertRaisesRegex(ValueError, "at least four"):
            apply_filtered_boundary_reconstruction(
                torch.ones(2, 3, 3), predicted, threshold=3.0
            )
        with self.assertRaisesRegex(ValueError, "predicted_last"):
            apply_filtered_boundary_reconstruction(
                batch, torch.ones(2, 1, 3), threshold=3.0
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            apply_filtered_boundary_reconstruction(
                batch, predicted, threshold=0.0
            )


if __name__ == "__main__":
    unittest.main()
