# CST205 Music Mixer

A PySide6 desktop stem mixer powered by Demucs. Upload an audio track, extract
vocals, drums, bass, guitar, piano, and other stems, then adjust each stem's
volume or mute state and play the resulting mix.

## Run

Using the existing CST205 environment from this repository directory:

```bash
../cst205env/bin/python main.py
```

Or create an environment and install the app dependency:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

The first separation downloads Demucs' `htdemucs_6s` model if it is not already
cached. CPU separation can take a few minutes; the app stays responsive while
it runs. The generated stems and current mix are stored in a temporary session
directory and removed by the operating system when the session ends.
