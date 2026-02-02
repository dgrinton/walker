# Walker - Termux Setup

## 1. Install Termux

Download from F-Droid (NOT Google Play - that version is outdated):
https://f-droid.org/en/packages/com.termux/

## 2. Install Termux:API

Also from F-Droid:
https://f-droid.org/en/packages/com.termux.api/

## 3. Setup Termux

Open Termux and run:

```bash
# Update packages
pkg update && pkg upgrade

# Install required packages
pkg install python termux-api espeak

# Install Python dependencies
pip install networkx requests

# Grant location permission (will prompt)
termux-location
```

## 4. Transfer walker.py to your phone

Options:
- Use `termux-setup-storage` then copy to ~/storage/downloads
- Use `git clone` if you have this in a repo
- Copy via USB

## 5. Run

```bash
# Navigate to where you put walker.py
cd ~/storage/downloads  # or wherever

# Run with default 2km walk
python walker.py

# Or specify distance in km
python walker.py 3.5  # for 3.5km walk
```

## 6. Connect Bluetooth headset

Pair your Bluetooth headset with your phone normally through Android settings.
Audio will route through it automatically.

## Troubleshooting

**"termux-location not found"**
- Install termux-api package: `pkg install termux-api`
- Install Termux:API app from F-Droid

**GPS timeout**
- Make sure location is enabled in Android settings
- Grant Termux location permission in Android settings
- Try going outside (GPS doesn't work well indoors)

**No audio**
- Install espeak: `pkg install espeak`
- Check Bluetooth is connected
- Check phone volume

**"Could not fetch map data"**
- Check internet connection
- The Overpass API might be overloaded, wait and try again
