# PD target torque recording

Single-environment `Extreme-RGMT-Stage2-G1-TheShy-Mushroom` playback automatically
records torque. The existing `play.py` command needs no extra flags. Files are
written to the repository's `logs/stage2/pd_target_torque_50hz_<timestamp>.csv`.
Every run creates a new file.

The logger reads Isaac Lab's `robot.data.computed_torque`: **unclipped PD target
torque estimates**, in N m, at the joint output side. These are not measurements
of the PhysX implicit drive output. Values can exceed actuator torque limits.
The existing controller, gains, randomization and disturbances are retained.

Each policy step arms one sample. After the controlled robot's first
`write_data_to_sim()` updates its actuator buffers, and before that substep is
integrated, the logger copies the 29 torques. The other three physics substeps
are skipped. Thus data is sampled at 50 Hz, with timestamps at the start of
each 0.02 s control interval. Reference-robot data is not recorded.

CSV columns:

- `episode_id`: 0, since only the first episode is recorded.
- `time_s`: simulation time elapsed since recording began, starting at zero.
- `motion_clip_id`: selected dataset clip ID.
- `motion_time_s`: reference-motion time at command submission.
- 29 joint-name columns, in `robot.joint_names` order, containing signed N m.

Recording and playback end on the first episode completion, Ctrl+C, window
closure, or an explicitly requested shorter duration/video length. Rows are
flushed as they are written; `finally` closes the file and restores the command
submission method on normal exit, interruption, or an error. A second forced
interrupt during shutdown is not needed.

With `--full-motions`, torque recording runs through the clip's actual duration.
The usual early motion timeout (the 0.2 s future-reference horizon) is disabled
only for this mode; out-of-range future reference queries use the sampler's
existing last-frame clamp. No torque is sampled at/after the subsequent reset.
The final sample represents the last control interval before clip completion,
so its timestamp is normally one policy step before the terminal frame.
An NPZ may contain several physical full circles; this mode records the entire
clip once, without detecting or splitting individual circles.

Options:

```text
--no-record-torques                  Restore continuous playback without CSV
--record-torques                     Explicitly enable for other compatible RGMT tasks
--torque-output-dir D:\some\folder   Override output directory
--torque-duration 2.0                Stop after 2 simulation seconds (or earlier episode end)
```

Recording requires one 29-joint robot and a 50 Hz policy step. `--real-time`
affects wall-clock pacing, not CSV timestamps or sample rate.

Lightweight recorder tests (no Isaac Sim startup required):

```powershell
python -m unittest discover -s scripts/rsl_rl -p test_torque_recorder.py -v
```
