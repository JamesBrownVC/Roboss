# Go2 WebRTC Control

Dashboard web pour contrôler le Unitree Go2 Pro via
[unitree_webrtc_connect](https://github.com/legion1581/unitree_webrtc_connect)
(même protocole WebRTC que l'app mobile Unitree — pas de jailbreak requis).

## Fonctionnalités

| Domaine | Détails |
|---|---|
| 🎮 Pilotage | D-pad + clavier (WASD, Q/E yaw, Shift boost, Espace stop), vitesse réglable |
| 🧍 Commandes sport | **Toutes** les `SPORT_CMD` / `SPORT_CMD_MCF` : postures, tricks (Hello, Dance…), gaits, acrobaties (flips, handstand…) + envoi libre avec paramètre JSON |
| ⚙️ Modes moteur | motion_switcher : normal / ai / advanced / mcf |
| 📷 Caméra | Flux vidéo temps réel (WebRTC → MJPEG) |
| 🌐 LiDAR | Activation + nuage de points vue de dessus |
| 🛡 Évitement d'obstacles | Toggle + drive routé via l'API obstacles_avoid |
| 💡 VUI | Couleur LED, luminosité, volume |
| 🔊 Audio Hub | Liste/lecture des fichiers audio, pause/reprise, mode lecture, mégaphone |
| 📊 Télémétrie | Batterie BMS (SOC, courant, températures), IMU (roll/pitch/yaw), 12 moteurs (temp/position/couple), forces des pattes, sport state (position, vitesse, obstacles) |
| 🛰 RPC générique | N'importe quel topic `RTC_TOPIC` + api_id + paramètre → réponse JSON brute |
| 📡 Topics bruts | Visualisation JSON en direct de tous les topics souscrits |

## Prérequis

- Mac relié au Go2 en ethernet, IP statique `192.168.123.99/24` (voir [../NETWORK.md](../NETWORK.md))
- Le robot répond sur `192.168.123.161`
- Python 3.10+
- **Firmware ≥ 1.1.15** : nécessite la clé AES-128 par appareil
  (`unitree-fetch-aes-key --email ... --device-type Go2`, compte Unitree requis)

## Installation

```bash
cd robotdog/webrtc-control
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> `portaudio` est requis par la lib : `brew install portaudio` si l'install échoue.

## Lancement

```bash
export UNITREE_ROBOT_IP=192.168.123.161        # défaut
# export UNITREE_AES_128_KEY=<32-hex>          # seulement firmware >= 1.1.15
python server.py
```

Puis ouvrir **http://localhost:8080**.

## Architecture

```
Browser (static/) ── WebSocket /ws ──┐
        │                            ├── server.py (aiohttp) ── WebRTC ── Go2
        └── MJPEG /video ────────────┘        unitree_webrtc_connect
```

- `server.py` : pont unique — connexion WebRTC au robot, souscription aux topics
  d'état (lowstate, sportmodestate, multiplestate, lidar_state, servicestate,
  audiohub, uwb, gas_sensor), ré-encodage vidéo en MJPEG, fan-out WebSocket
  throttlé à 5 Hz, dispatch des commandes.
- `static/` : UI vanilla JS (aucun build), génère les boutons à partir du dump
  `/api/constants` (donc automatiquement à jour avec la lib).

## Notes

- Un seul client WebRTC à la fois côté robot : fermer l'app mobile Unitree
  avant de lancer le serveur (sinon `RobotBusyError`).
- Les acrobaties (flips…) exigent le mode **ai**/**mcf** et de l'espace libre.
- En mode `mcf`, les api_id `SPORT_CMD_MCF` sont utilisés automatiquement.
