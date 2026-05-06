# Driver Fatigue Detection System

## Objective
This project implements a computer vision system for driver fatigue detection with gesture-based activation.

## Features
- System starts inactive
- Activates only after a correct gesture sequence
- Detects fatigue using facial visual cues

## Setup
1. Create a virtual environment
2. Install requirements
3. Run the main script

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python src/main.py
```

## Controls
| Key | Action |
|-----|--------|
| `1` | Set state to INACTIVE |
| `2` | Set state to WAITING_FOR_SEQUENCE |
| `3` | Set state to ACTIVE |
| `4` | Set state to ALERT |
| `q` | Quit |

## Team
- Member 1: Gesture activation
- Member 2: Eye closure detection
- Member 3: Yawning and head pose
- Member 4: Integration and report
