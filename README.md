# RL Super Mario Bros (PPO)

Tränar en agent att spela **Super Mario Bros (NES)** med **Proximal Policy Optimization (PPO)** via `stable-baselines3` och `gym-retro` (`retro`).

Projektet använder:
- parallella environments via `SubprocVecEnv` (snabbare träning)
- `VecMonitor`/`Monitor`-loggar för reward-statistik
- TensorBoard-loggning
- callback som sparar bästa modellen baserat på mean reward

## Demo / Status
- Träningsscript: `train.py` (eller `run.py` beroende på vad du använder)
- Loggar hamnar i `tmp/` och TensorBoard i `board/`

## ROM / gym-retro setup (viktig)

`gym-retro` kräver att spelet är importerat i Retro-databasen.
```bash

python -m retro.import /path/to/roms

> **Obs:** ROM-filer ingår inte i repot.

---

## Krav
- Python 3.9+ (3.10/3.11 brukar funka bra)
- Windows/Linux/macOS
- En NES-ROM av Super Mario Bros (laglig kopia) som krävs av `gym-retro`

Bibliotek:
- `stable-baselines3`
- `gym-retro` / `retro`
- `numpy`
- (rekommenderat) `tensorboard`

---

## Installation

### 1) Skapa venv
```bash
python -m venv venv
# Windows (PowerShell)
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
