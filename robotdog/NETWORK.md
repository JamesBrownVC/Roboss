# Go2 Pro — Connexion réseau & SSH

## Setup physique

- MacBook Pro connecté **directement en Ethernet** au Go2 Pro via un adaptateur USB (`USB 10/100/1000 LAN`, interface `en9`).
- Le Go2 utilise le sous-réseau statique **192.168.123.0/24** (pas de DHCP côté robot).

## Configuration réseau du Mac (effectuée le 2026-07-04)

Sans IP manuelle, l'adaptateur récupère une adresse self-assigned (169.254.x.x) et le robot est injoignable.

```bash
sudo networksetup -setmanual "USB 10/100/1000 LAN" 192.168.123.99 255.255.255.0
```

| Paramètre | Valeur |
|---|---|
| Interface | `en9` (USB 10/100/1000 LAN) |
| IP Mac | `192.168.123.99` |
| Masque | `255.255.255.0` |
| Routeur | aucun (liaison directe) |

Équivalent GUI : Réglages Système → Réseau → USB 10/100/1000 LAN → Détails → TCP/IP → Configurer IPv4 : *Manuellement*.

## Adresses du robot

Scan du sous-réseau : seule **192.168.123.161** répond (MAC `7e:1d:75:60:f5:89`).
Les adresses classiques Unitree (.18, .13, .14, .15, .162, .163) ne répondent pas sur ce Go2 Pro.

## Connexion SSH

```bash
ssh root@192.168.123.161
```

- Utilisateur : `root`
- Mot de passe : `theroboverse`

(Le compte `unitree` / `123` par défaut existe aussi mais le compte utilisé est `root`.)

## Vérifications rapides

```bash
ifconfig en9 | grep "inet "        # doit afficher 192.168.123.99
ping -c 2 192.168.123.161          # le robot doit répondre
nc -z -G 3 192.168.123.161 22      # port SSH ouvert
```

## Outils firmware embarqués

Sur le robot, dans `/unitree/dev/go2_firmware_tools` :

```
README.md  downloads  firmware  main.py  requirements.txt  update.sh
device     files      install.sh  network  start.sh        util
```

Lancement du menu interactif :

```bash
cd /unitree/dev/go2_firmware_tools
./start.sh
# Options : Device / Firmware / Network / Quit
```
