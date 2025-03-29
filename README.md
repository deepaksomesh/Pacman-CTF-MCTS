# Monte Carlo Tree Search (MCTS) for Pacman-CTF

## Overview
In this project, we implement **Monte Carlo Tree Search (MCTS)** on the **Pacman-CTF agents** to make optimal decisions to win the game. We have also explored a few improvements, including:
- **Rapid Value Action Estimation (RAVE)**: Enhances early exploration.
- **Heavy Playout**: Uses an evaluation function to improve rollouts.

## Implemented Agents

### **1. MCTSTeam.py**
- Contains the standard **MCTS implementation**.
- Performs **simple heuristic actions**.

### **2. RaveTeam.py**
- Implements **RAVE (Rapid Value Action Estimation)**.
- Boosts **exploration in early simulations**.

### **3. HeavyPlayoutTeam.py**
- The agent repeatedly selects actions from the **starting state**.
- Applies actions to the game state until the **terminal condition** is reached.
- After the rollout, it evaluates the resulting state using an **evaluation function**.

## Setup
Before running the code, install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running a Match
Use the following command to conduct a match between two agents:
```bash
python capture.py --red=<filename> --blue=<filename>
```
Replace `<filename>` with the appropriate agent script (e.g., `MCTSTeam.py`).

## Running a Tournament
To run a tournament using the **Elo rating system**, execute:
```bash
python elo.py
```

---

Feel free to modify and experiment with the agents to improve their performance!

