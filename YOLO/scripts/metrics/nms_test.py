import unittest
from metrics.nms import nonMaximalSuppression


class TestNonMaxSuppression(unittest.TestCase):
    def setUp(self):
        # test cases we want to run
        self.t1_boxes = [
            [1, 1, 0.5, 0.45, 0.4, 0.5],
            [1, 0.8, 0.5, 0.5, 0.2, 0.4],
            [1, 0.7, 0.25, 0.35, 0.3, 0.1],
            [1, 0.05, 0.1, 0.1, 0.1, 0.1],
        ]

        self.c1_boxes = [[1, 1, 0.5, 0.45, 0.4, 0.5], [1, 0.7, 0.25, 0.35, 0.3, 0.1]]

        self.t2_boxes = [
            [1, 1, 0.5, 0.45, 0.4, 0.5],
            [2, 0.9, 0.5, 0.5, 0.2, 0.4],
            [1, 0.8, 0.25, 0.35, 0.3, 0.1],
            [1, 0.05, 0.1, 0.1, 0.1, 0.1],
        ]

        self.c2_boxes = [
            [1, 1, 0.5, 0.45, 0.4, 0.5],
            [2, 0.9, 0.5, 0.5, 0.2, 0.4],
            [1, 0.8, 0.25, 0.35, 0.3, 0.1],
        ]

        self.t3_boxes = [
            [1, 0.9, 0.5, 0.45, 0.4, 0.5],
            [1, 1, 0.5, 0.5, 0.2, 0.4],
            [2, 0.8, 0.25, 0.35, 0.3, 0.1],
            [1, 0.05, 0.1, 0.1, 0.1, 0.1],
        ]

        self.c3_boxes = [[1, 1, 0.5, 0.5, 0.2, 0.4], [2, 0.8, 0.25, 0.35, 0.3, 0.1]]

        self.t4_boxes = [
            [1, 0.9, 0.5, 0.45, 0.4, 0.5],
            [1, 1, 0.5, 0.5, 0.2, 0.4],
            [1, 0.8, 0.25, 0.35, 0.3, 0.1],
            [1, 0.05, 0.1, 0.1, 0.1, 0.1],
        ]

        self.c4_boxes = [
            [1, 0.9, 0.5, 0.45, 0.4, 0.5],
            [1, 1, 0.5, 0.5, 0.2, 0.4],
            [1, 0.8, 0.25, 0.35, 0.3, 0.1],
        ]

    def test_remove_on_iou(self):
        bboxes = nonMaximalSuppression(
            self.t1_boxes,
            probThreshold=0.2,
            iouThreshold=7 / 20,
            boxFormat="midpoint",
        )
        self.assertTrue(sorted(bboxes) == sorted(self.c1_boxes))

    def test_keep_on_class(self):
        bboxes = nonMaximalSuppression(
            self.t2_boxes,
            probThreshold=0.2,
            iouThreshold=7 / 20,
            boxFormat="midpoint",
        )
        self.assertTrue(sorted(bboxes) == sorted(self.c2_boxes))

    def test_remove_on_iou_and_class(self):
        bboxes = nonMaximalSuppression(
            self.t3_boxes,
            probThreshold=0.2,
            iouThreshold=7 / 20,
            boxFormat="midpoint",
        )
        self.assertTrue(sorted(bboxes) == sorted(self.c3_boxes))

    def test_keep_on_iou(self):
        bboxes = nonMaximalSuppression(
            self.t4_boxes,
            probThreshold=0.2,
            iouThreshold=9 / 20,
            boxFormat="midpoint",
        )
        self.assertTrue(sorted(bboxes) == sorted(self.c4_boxes))


if __name__ == "__main__":
    print("Running Non Max Suppression Tests:")
    unittest.main()
