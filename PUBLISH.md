# Activate env -

```bash
source pelican-env/bin/activate  # On macOS/Linux
pelican-env\Scripts\activate  # On Windows
```
## Dependencies
```bash
pip install pelican markdown beautifulsoup4
```

## Local Preview
```bash
pelican --listen --autoreload
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) to view site contents locally.

## Generate content
```bash
pelican content -s pelicanconf.py
```
