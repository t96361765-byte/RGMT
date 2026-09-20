"""CSV logging of first-substep PD target torque, without changing actuator commands.

For ImplicitActuator, computed_torque is an unclipped PD estimate, not a
measurement of the PhysX drive output. All rotary joint values are in N m.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


class TorqueCsvRecorder:
    """Observe one robot's command submission once per explicitly armed policy step.

The temporary instance hook runs AFTER Isaac Lab updates computed_torque in
write_data_to_sim, BEFORE the first physics integration of the policy step.
Further substeps and automatic-reset command writes do not produce samples.
The original method is restored on close, including on KeyboardInterrupt.
"""

    def __init__(self, robot, output_dir: str | Path):
        self.robot = robot
        self.joint_names = list(robot.joint_names)
        if robot.num_instances != 1 or len(self.joint_names) != 29:
            raise ValueError("Torque CSV recording requires one robot with exactly 29 joints.")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.path = output_dir / f"pd_target_torque_50hz_{stamp}.csv"
        self._file = self.path.open("x", encoding="utf-8", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["episode_id", "time_s", "motion_clip_id", "motion_time_s", *self.joint_names])
        self._file.flush()
        self.rows_written = 0
        self._pending = None
        self._closed = False
        self._original_write = robot.write_data_to_sim
        self._had_instance_write = "write_data_to_sim" in vars(robot)
        self._instance_write = vars(robot).get("write_data_to_sim")

        def write_and_record(*args, **kwargs):
            result = self._original_write(*args, **kwargs)
            if self._pending is not None:
                values = self.robot.data.computed_torque[0].detach().cpu().tolist()
                if len(values) != len(self.joint_names):
                    raise RuntimeError("Joint torque count changed during playback.")
                self._writer.writerow([*self._pending, *values])
                # Keep completed rows available even when playback is interrupted.
                self._file.flush()
                self.rows_written += 1
                self._pending = None
            return result

        robot.write_data_to_sim = write_and_record

    def begin_step(self, time_s: float, motion_clip_id: int, motion_time_s: float) -> None:
        """Arm the first command submission; timestamps describe the pre-step state."""
        if self._closed or self._pending is not None:
            raise RuntimeError("Recorder is closed or the previous policy step was not sampled.")
        self._pending = (0, f"{time_s:.6f}", motion_clip_id, f"{motion_time_s:.6f}")

    def end_step(self) -> None:
        """Fail visibly if an incompatible environment never submits robot commands."""
        if self._pending is not None:
            raise RuntimeError("No robot command submission was observed during env.step().")

    def close(self) -> None:
        if self._closed:
            return
        if self._had_instance_write:
            self.robot.write_data_to_sim = self._instance_write
        else:
            del self.robot.write_data_to_sim
        self._pending = None
        self._file.close()
        self._closed = True
