## Introduction
my Extreme-RGMT training for stage_1 and stage_2, including robot-only and robot-mushroom tasks, pure motion tracking

## Extreme-RGMT Stage1

### headless from zero

```powershell
& D:\miniconda3\envs\env_isaaclab\python.exe `
  .\scripts\rsl_rl\train.py `
  --task Isaac-RGMT-G1-v0 `
  --motion-file "D:\track_dataset\lafan1_npz" `
  --num_envs 4096 `
  --max_iterations 100000 `
  --run_name RGMT_lafan1_npz `
  --headless
```

### continue training

```powershell
& D:\miniconda3\envs\env_isaaclab\python.exe `
  .\scripts\rsl_rl\train.py `
  --task Isaac-RGMT-G1-v0 `
  --motion-file "D:\track_dataset\lafan1_npz" `
  --num_envs 4096 `
  --resume `
  --load_run "2026-08-19_21-25-55_RGMT_lafan1_npz_continue2" `
  --checkpoint "model_125000.pt" `
  --max_iterations 25000 `
  --run_name "RGMT_lafan1_npz_continue3" `
  --headless
```

## Extreme-RGMT Stage2 

### robot-only Train

```powershell
& D:\miniconda3\envs\env_isaaclab\python.exe `
  .\scripts\rsl_rl\train.py `
  --task Isaac-Extreme-RGMT-Stage2-G1-v0 `
  --motion-file "D:\track_dataset\Extreme-RGMT_50Hz" `
  --mastered-motion-file "D:\track_dataset\lafan1_npz" `
  --stage1-checkpoint "D:\GitHub\RGMT\logs\rsl_rl\extreme_rgmt_stage1_g1\2026-08-20_21-59-33_RGMT_lafan1_npz_continue3\model_149999.pt" `
  --num_envs 4096 `
  --max_iterations 50000 `
  --run_name "RGMT_stage2" `
  --headless
```

### robot-only Coutinue Train

```powershell
& D:\miniconda3\envs\env_isaaclab\python.exe `
  .\scripts\rsl_rl\train.py `
  --task Isaac-Extreme-RGMT-Stage2-G1-v0 `
  --motion-file "D:\track_dataset\Extreme-RGMT_50Hz" `
  --mastered-motion-file "D:\track_dataset\lafan1_npz" `
  --num_envs 4096 `
  --resume `
  --load_run "2026-09-01_23-18-30_RGMT_stage2" `
  --checkpoint "model_45000.pt" `
  --max_iterations 55000 `
  --run_name "RGMT_stage2_continue1" `
  --headless
```

### robot-mushroom train

```powershell
Set-Location D:\GitHub\RGMT
& D:\miniconda3\envs\env_isaaclab\python.exe `
  .\scripts\rsl_rl\train.py `
  --task Extreme-RGMT-Stage2-G1-TheShy-Mushroom `
  --motion-file "D:\track_dataset\Flare_50Hz" `
  --mastered-motion-file "D:\track_dataset\lafan1_npz" `
  --stage1-checkpoint "D:\GitHub\RGMT\logs\rsl_rl\extreme_rgmt_stage1_g1\2026-08-20_21-59-33_RGMT_lafan1_npz_continue3\model_149999.pt" `
  --num_envs 2048 `
  --max_iterations 30000 `
  --run_name "flare-v1" `
  --seed 42 `
  --headless
```

### robot-mushroom continue train

```powershell
Set-Location D:\GitHub\RGMT
& D:\miniconda3\envs\env_isaaclab\python.exe `
  .\scripts\rsl_rl\train.py `
  --task Extreme-RGMT-Stage2-G1-TheShy-Mushroom `
  --motion-file "D:\track_dataset\Flare_50Hz" `
  --mastered-motion-file "D:\track_dataset\lafan1_npz" `
  --num_envs 2048 `
  --resume `
  --load_run "2026-09-08_19-30-23_flare-v1_continue1" `
  --checkpoint "model_59998.pt" `
  --max_iterations 40000 `
  --run_name "flare-v1_continue2" `
  --seed 42 `
  --headless
```

## IsaacSim Play Extreme-RGMT

### robot-only 

```powershell
Set-Location D:\GitHub\RGMT
& D:\miniconda3\envs\env_isaaclab\python.exe `
  .\scripts\rsl_rl\play.py `
  --task Isaac-Extreme-RGMT-Stage2-G1-v0 `
  --motion-file "D:\track_dataset\Extreme-RGMT_50Hz\lt_cc3_50Hz.npz" `
  --motion-clip-id 0 `
  --motion-start-time 0.0 `
  --full-motions `
  --disable-early-termination `
  --num_envs 1 `
  --checkpoint "D:\GitHub\RGMT\logs\rsl_rl\extreme_rgmt_stage2_g1\2026-09-03_22-35-10_RGMT_stage2_continue1\model_99999.pt" `
  --seed 42 `
  --real-time `
  --reference-robot `
  --reference-offset-y 2
```

### robot-mushroom 

```powershell
Set-Location D:\GitHub\RGMT
python `
  .\scripts\rsl_rl\play.py `
  --task Extreme-RGMT-Stage2-G1-TheShy-Mushroom `
  --motion-file "D:\track_dataset\Flare_50Hz\flare_mushroom_0.85_50Hz.npz" `
  --motion-clip-id 0 `
  --motion-start-time 0.0 `
  --full-motions `
  --disable-early-termination `
  --num_envs 1 `
  --checkpoint "D:\GitHub\RGMT\logs\stage2\model_28000.pt" `
  --seed 42 `
  --real-time `
  --reference-robot `
  --reference-offset-y 2
```

## MuJoCo Deploy Extreme-RGMT

### robot-only

```powershell
### policy.onnx
& D:\miniconda3\envs\env_isaaclab\python.exe `
  .\scripts\rsl_rl\play_mujoco.py `
  --policy "D:\GitHub\RGMT\logs\rsl_rl\extreme_rgmt_stage2_g1\2026-09-03_22-35-10_RGMT_stage2_continue1\exported\policy.onnx" `
  --motion-file "D:\track_dataset\Extreme-RGMT_50Hz\CMU_90_08_50Hz.npz" `
  --device cuda `
  --real-time
```

```powershell
### policy.pt
& D:\miniconda3\envs\env_isaaclab\python.exe `
  .\scripts\rsl_rl\play_mujoco.py `
  --policy "D:\GitHub\RGMT\logs\rsl_rl\extreme_rgmt_stage2_g1\2026-09-01_23-18-30_RGMT_stage2\exported\policy.pt" `
  --motion-file "D:\track_dataset\lafan1_npz\run1_subject2.npz" `
  --device cuda `
  --real-time
```

### robot-mushroom

```powershell
Set-Location D:\GitHub\RGMT
& D:\miniconda3\envs\env_isaaclab\python.exe `
  .\scripts\rsl_rl\play_mujoco.py `
  --policy "D:\GitHub\RGMT\logs\rsl_rl\extreme_rgmt_stage2_g1\2026-09-08_19-30-23_flare-v1_continue1\exported\policy.pt" `
  --robot-model "D:\GitHub\RGMT\source\RGMT\data\Robots\G1\g1_theshy\g1_theshy_mushroom.urdf" `
  --motion-file "D:\track_dataset\Flare_50Hz\flare_mushroom_0.85_50Hz.npz" `
  --mushroom `
  --mushroom-scale 0.85 `
  --device cuda `
  --real-time
```

## Tensorboard 

```powershell
Set-Location D:\GitHub\RGMT
& D:\miniconda3\envs\env_isaaclab\python.exe `
  -m tensorboard.main `
  --logdir "D:\GitHub\RGMT\logs\rsl_rl\extreme_rgmt_stage2_g1\2026-09-10_01-05-57_flare-v1_continue2" `
  --port 6006
```
