"""Run without Isaac Sim: python -m unittest discover -s scripts/rsl_rl -p test_torque_recorder.py."""

import csv
import tempfile
import unittest
from types import SimpleNamespace

from torque_recorder import TorqueCsvRecorder


class FakeRow:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)


class FakeRobot:
    num_instances = 1
    joint_names = [f"joint_{index}" for index in range(29)]

    def __init__(self):
        self.writes = 0
        self.data = SimpleNamespace(computed_torque=[FakeRow([-999] * 29)])

    def write_data_to_sim(self):
        self.writes += 1
        self.data.computed_torque = [FakeRow([self.writes + index / 100 for index in range(29)])]


class TorqueCsvRecorderTests(unittest.TestCase):
    def test_first_substep_not_stale_or_last_substep_and_ignore_reset(self):
        robot = FakeRobot()
        with tempfile.TemporaryDirectory() as directory:
            recorder = TorqueCsvRecorder(robot, directory)
            try:
                for step in range(3):
                    recorder.begin_step(step * 0.02, 7, 1.0 + step * 0.02)
                    for _ in range(4):
                        robot.write_data_to_sim()
                    recorder.end_step()
                robot.write_data_to_sim()  # Automatic reset must not add a sample.
                with recorder.path.open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))  # Rows are readable before close.
                self.assertEqual(len(rows), 3)
                self.assertEqual(len(rows[0]), 33)
                self.assertEqual([float(row["joint_0"]) for row in rows], [1, 5, 9])
                self.assertEqual([float(row["time_s"]) for row in rows], [0, 0.02, 0.04])
                self.assertEqual([float(row["motion_time_s"]) for row in rows], [1, 1.02, 1.04])
                self.assertEqual(rows[0]["motion_clip_id"], "7")
                self.assertAlmostEqual(float(rows[0]["joint_28"]), 1.28)
            finally:
                recorder.close()
            self.assertNotIn("write_data_to_sim", vars(robot))
            robot.write_data_to_sim()
            self.assertEqual(robot.writes, 14)

    def test_interrupt_keeps_partial_csv_and_restores_hook(self):
        robot = FakeRobot()
        with tempfile.TemporaryDirectory() as directory:
            recorder = TorqueCsvRecorder(robot, directory)
            try:
                recorder.begin_step(0, 0, 0)
                robot.write_data_to_sim()
                raise KeyboardInterrupt
            except KeyboardInterrupt:
                pass
            finally:
                recorder.close()
            recorder.close()  # Cleanup is idempotent.
            with recorder.path.open(newline="", encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)
            self.assertNotIn("write_data_to_sim", vars(robot))

    def test_missing_submission_and_multiple_environments_are_rejected(self):
        robot = FakeRobot()
        with tempfile.TemporaryDirectory() as directory:
            recorder = TorqueCsvRecorder(robot, directory)
            try:
                recorder.begin_step(0, 0, 0)
                with self.assertRaises(RuntimeError):
                    recorder.end_step()
            finally:
                recorder.close()
            robot.num_instances = 2
            with self.assertRaises(ValueError):
                TorqueCsvRecorder(robot, directory)


if __name__ == "__main__":
    unittest.main()
