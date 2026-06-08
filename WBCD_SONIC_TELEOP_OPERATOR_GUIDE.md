# WBCD Sonic 比赛遥操作流程

本文档用于 WBCD Track 1 logistics picking 比赛现场操作。当前方案中，视频显示在 Ubuntu 宿主机屏幕上；PICO 只负责 Sonic 全身遥操作和手部控制。

## 1. 系统启动

### 机器人端 deploy

在机器人/部署端启动 Sonic deploy，使用 `zmq_manager` 输入：

```bash
cd /home/long/workspace_wbcd_sonic/GR00T-WholeBodyControl/gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type zmq_manager
```

如果 deploy 端和 PICO manager 不在同一台机器，按实际网络增加 `--zmq-host <PICO_MANAGER_IP>`。

### PICO manager 端

在运行 PICO/XRoboToolkit 的机器上启动：

```bash
cd /home/long/workspace_wbcd_sonic/GR00T-WholeBodyControl
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py --manager \
  --wbcd_competition_config gear_sonic/config/wbcd_pico_competition.yaml
```

配置文件支持热更新：

```text
gear_sonic/config/wbcd_pico_competition.yaml
```

比赛现场可改半蹲高度、恢复速度、低速移动速度等参数，不需要重启程序。

## 2. 模式定义

| 模式 | 实际 Sonic StreamMode | 用途 |
|---|---|---|
| `OFF` | `OFF` | 策略未启动或急停 |
| `TRANSPORT` | `POSE` | 原始 Sonic 全身 POSE 遥操作，用于走到货架和搬运物体 |
| `STAND_MANIP` | `PLANNER_VR_3PT` | 站姿抓取，上层货架优先使用 |
| `HALF_SQUAT_MANIP` | `PLANNER_VR_3PT + IDLE_SQUAT` | 半蹲抓取，中层货架优先使用 |
| `KNEEL_MANIP` | `PLANNER_VR_3PT + IDLE_KNEEL_TWO_LEGS` | 双膝跪姿抓取，下层货架谨慎使用 |
| `STAND_RECOVERY` | `PLANNER_VR_3PT` | 从抓取姿态缓慢恢复站立 |

重点：`TRANSPORT` 不是只控制双臂的 VR_3PT，而是原始全身 `POSE` 模式。

## 3. 按键表

### 原 Sonic 基础按键

| 操作 | 按键 | 说明 |
|---|---|---|
| 启动/停止策略 | `A+B+X+Y` | 第一次启动并做全局校准；再次按为停止/急停 |
| 进入/退出全身 POSE | `A+X` | 在 `PLANNER` 和 `POSE/TRANSPORT` 间切换 |
| 进入/退出 frozen upper planner | `B+Y` | 保留原 Sonic 功能 |
| 进入/退出普通 VR_3PT | 左摇杆按下 | 保留原 Sonic 功能 |
| 手部抓取 | 左/右 Trigger 或 Grip | 控制对应手部闭合 |

### WBCD 比赛新增按键

| 操作 | 按键 | 说明 |
|---|---|---|
| 进入站姿抓取 | `left_menu + A` | 从 `POSE/TRANSPORT` 切到 `STAND_MANIP` |
| 进入半蹲抓取 | `left_menu + X` | 从 `POSE/TRANSPORT` 切到 `HALF_SQUAT_MANIP` |
| 进入跪姿抓取 | `left_menu + Y` | 只能先进入 WBCD 抓取模式；推荐 `HALF_SQUAT_MANIP` 后再进入 |
| 半蹲升高 | `Y` | 仅在 `HALF_SQUAT_MANIP` 且不按 `left_menu` 时生效 |
| 半蹲降低 | `X` | 仅在 `HALF_SQUAT_MANIP` 且不按 `left_menu` 时生效 |
| 跪姿升高 | `Y` | 仅在 `KNEEL_MANIP` 且不按 `left_menu` 时生效；每按一次升高一档 |
| 跪姿降低 | `X` | 仅在 `KNEEL_MANIP` 且不按 `left_menu` 时生效；每按一次降低一档 |
| 缓慢恢复站立 | 按住 `left_menu + B` | 半蹲/站姿普通恢复；跪姿分阶段恢复；松开暂停 |
| 从 WBCD 回全身 POSE | `A+X` 或 `B+Y` | 退出 WBCD 抓取模式，回到 `POSE/TRANSPORT` |
| 急停 | `A+B+X+Y` | 任意模式下停止 |

## 4. 推荐比赛流程

### Step 1：启动与校准

1. 操作员站到 Sonic 校准姿势。
2. 按 `A+B+X+Y` 启动策略。
3. 按 `A+X` 进入 `POSE/TRANSPORT`。
4. 确认机器人能跟随全身动作，摇杆能控制前后移动和 yaw 转向。

### Step 2：移动到货架前

1. 保持 `POSE/TRANSPORT`。
2. 操作员只做前后移动，机器人转向主要用摇杆 yaw 控制。
3. 根据 Ubuntu 端视频画面，把机器人停到适合抓取的货架距离。

### Step 3：根据货架高度切抓取模式

上层物体：

1. 在货架前保持身体稳定。
2. 按 `left_menu + A` 进入 `STAND_MANIP`。
3. 用 VR_3PT 控制双臂和双手抓取。

中层物体：

1. 在货架前保持身体稳定。
2. 先把双手放到安全位置，再按 `left_menu + X` 进入 `HALF_SQUAT_MANIP`。
3. 程序会用机器人当前实测手臂姿态做一次连续标定，随后短暂保持站姿，再缓慢进入半蹲。
4. 如高度不合适：
   - 按住 `Y` 升高。
   - 按住 `X` 降低。
5. 用 VR_3PT 控制双臂和双手抓取。

下层物体：

1. 先按 `left_menu + X` 进入 `HALF_SQUAT_MANIP`。
2. 确认机器人稳定，并且操作者双手已经放到安全位置。
3. 再按 `left_menu + Y` 进入 `KNEEL_MANIP`。
4. 跪姿是静态模式，不建议移动；只用 VR_3PT 控制双臂和双手抓取。

安全限制：在 `POSE/TRANSPORT` 下直接按 `left_menu + Y` 不会进入跪姿，只会提示先进入 `HALF_SQUAT_MANIP`。

### Step 4：抓到物体后恢复

1. 抓稳物体后，先尽量把手和物体从货架内部移出。
2. 如果需要缓慢站起，按住 `left_menu + B`。
3. 如果从 `KNEEL_MANIP` 恢复，程序会按“双膝跪保持 -> 单膝跪 -> 半蹲 -> 缓慢站立”执行。
4. 如果发现可能撞货架，松开 `B` 暂停站起。
5. 再次按住 `left_menu + B` 会从当前阶段继续恢复。
6. 首次测试跪姿恢复时，务必远离货架确认单膝跪阶段稳定。
7. 完成恢复后，程序会回到 `POSE/TRANSPORT`。

### Step 5：运输与放置

1. 使用 `POSE/TRANSPORT` 搬运物体。
2. 到达箱子前后，用全身 POSE 和手部 Trigger/Grip 放置物体。
3. 重复 Step 2 到 Step 5。

## 5. 现场参数调整

编辑：

```bash
gear_sonic/config/wbcd_pico_competition.yaml
```

常用参数：

```yaml
height:
  half_squat_default: 0.55
  half_squat_min: 0.45
  half_squat_max: 0.62
  kneel_default: 0.50
  kneel_min: 0.30
  kneel_max: 0.70
  kneel_adjust_step_m: 0.02
  adjust_speed_mps: 0.06
  stand_recovery_target: 0.74
  stand_recovery_speed_mps: 0.10

movement:
  stand_manip_max_vx: 0.10
  stand_manip_max_wz: 0.25
  stand_recovery_max_vx: 0.08
  stand_recovery_max_wz: 0.25

entry:
  require_robot_feedback: true
  hold_before_squat_sec: 0.30
  squat_enter_speed_mps: 0.08
  print_vr_target_debug: false

recovery:
  kneel_hold_sec: 0.30
  one_kneel_height: 0.50
  one_kneel_hold_sec: 0.60
  squat_recovery_height: 0.58
  squat_hold_sec: 0.50
  squat_to_stand_speed_mps: 0.06
  allow_recovery_motion: false
```

调整建议：

| 现象 | 优先调整 |
|---|---|
| 半蹲太高 | 降低 `half_squat_default` |
| 半蹲太低或不稳 | 提高 `half_squat_default` 或 `half_squat_min` |
| 跪姿太低或不稳 | 提高 `kneel_default` 或 `kneel_min` |
| 跪姿单次调整太大/太小 | 调整 `kneel_adjust_step_m` |
| 进入半蹲太快 | 降低 `squat_enter_speed_mps` 或增加 `hold_before_squat_sec` |
| 按抓取模式无反应 | 检查终端是否提示缺少 robot feedback |
| 站起太快 | 降低 `stand_recovery_speed_mps` |
| 站起太慢 | 提高 `stand_recovery_speed_mps` |
| 跪姿恢复前冲 | 保持 `allow_recovery_motion: false`，降低 `squat_to_stand_speed_mps` |
| 单膝跪阶段不稳 | 提高 `one_kneel_height`，或缩短 `one_kneel_hold_sec` |
| 站姿抓取移动太快 | 降低 `stand_manip_max_vx` |
| 恢复时后退太快 | 降低 `stand_recovery_max_vx` |

## 6. 注意事项

- 从 `POSE/TRANSPORT` 进入 `STAND_MANIP` 或 `HALF_SQUAT_MANIP` 时，WBCD 会用机器人当前实测手臂姿态做连续标定；如果没有 deploy feedback，默认不会进入抓取模式。
- 在 WBCD 抓取模式内部切换 `STAND_MANIP`、`HALF_SQUAT_MANIP` 或 `KNEEL_MANIP` 时，不重新做全局零姿态标定；切换前先把操作者双手放到安全位置。
- `HALF_SQUAT_MANIP` 使用 deploy 端的 `IDLE_SQUAT`，该模式在当前 Sonic deploy 里属于 static mode，因此不保证半蹲状态下移动。
- `KNEEL_MANIP` 使用双膝跪姿 static mode，默认不移动；从 `POSE/TRANSPORT` 直接进入跪姿被禁止。
- 如果 WBCD 模式下操作异常，优先按 `A+X` 回到 `POSE/TRANSPORT`；紧急情况按 `A+B+X+Y`。
- 比赛时建议一名操作员负责 PICO，一名安全员盯 deploy 终端和机器人本体。
