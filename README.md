# ODS – selectable object detection from any camera

Open the page on a laptop or phone, tick the classes you care about (e.g. `person`, `car`, or `drone`),
press **Start**. Only the ticked classes are detected and drawn; changing the selection takes effect live.

```
browser camera (getUserMedia) ──JPEG frames──▶ FastAPI /ws/detect ──▶ YOLO (class-filtered) ──▶ boxes ──▶ canvas overlay
```

## How it works

### The pieces

| Piece | What it is | Where it runs |
|---|---|---|
| **Server** (`ods/server.py`) | FastAPI app: serves the web pages, the class list, and WebSockets for frames and pairing | Your PC |
| **Detector** (`ods/detector.py`) | Loads the YOLO weights and runs inference, keeping only the classes you ticked | Your PC (the same process) |
| **Model weights** (`yolo11m.pt`) | A pretrained YOLO11-medium file, about 39 MB | A file in the project folder on your PC |
| **Hub** (`ods/hub.py`) | In-memory matchmaker between laptops and phones, plus the frame relay | Your PC |
| **Laptop page** (`static/index.html`) | Class picker, local camera, live view with boxes, accept popup | Browser |
| **Phone page** (`static/phone.html`) | Lists nearby computers and streams the phone's camera | Browser |
| **Tunnel** (`ods/share.py`, only with `--share`) | Gives your PC a public HTTPS address | `cloudflared` on your PC |

### The model: trained already, and run locally

- `yolo11m.pt` is **already trained** by Ultralytics on the COCO dataset (80 everyday object classes). Nothing
  is trained when you run the app, and no model is downloaded at runtime.
- The weights file sits in the project folder (Ultralytics fetched it once the first time it was needed) and
  inference runs **in the server process on your own computer**, on a GPU if PyTorch finds one, otherwise
  on the CPU. No cloud AI service or API is called.
- It can only detect what it was trained on. The UI offers 11 COCO classes (person, car, dog, ...). `drone`
  is not a COCO class, so it stays greyed out until you add drone weights at `weights/drone.pt`.
- The server loads and warms the models in the background at start-up (about 35 s the first time), so the
  first frame isn't slow once it says it is ready.
- Roughly 0.5-1 s per frame on a CPU with the medium model. Use `yolo11s.pt` or `yolo11n.pt` for more speed.

### What happens to a frame

```
phone camera ──JPEG──▶ server ──decode──▶ YOLO (only the laptop's selected classes)
                          │                       │
                          └── relay JPEG + boxes ◀┘──▶ laptop draws the image and the boxes
```

1. The phone grabs a frame from its camera, downsizes it to at most 960 px wide, encodes it as JPEG and sends
   it over a WebSocket (`/ws/phone`). It sends the next frame only after the server acknowledges the last one,
   so slow hosts are never flooded.
2. The server decodes the JPEG and runs the model for the classes the **laptop** has ticked and its confidence
   setting. Boxes are normalised to 0-1, so any screen size can draw them.
3. The server forwards the JPEG together with the boxes to the laptop (`/ws/host`), which paints the image and
   overlays the boxes. When a computer uses its own camera instead of a phone, the page sends frames to
   `/ws/detect` and gets the boxes back directly; the model step is identical.

### Finding a computer and pairing

1. A laptop that ticked **Accept phone cameras** registers with the hub and is given a 6-digit code.
2. A phone asks `/api/hosts` for computers. The server lists the ones whose network address (public IP as it
   sees it) matches the phone's, longest-waiting first, or looks one up by code when you type it.
3. Tapping a computer sends a request. The laptop shows an Accept / Decline popup; nothing is streamed
   before it accepts. One phone per computer: other requests are told it is busy.
4. Either side can end the session at any time; closing a page also ends it.

### Sharing with `--share`: what Cloudflare is doing

- `uv run ods --share` uses a **Cloudflare Quick Tunnel** (TryCloudflare) through the open-source `cloudflared`
  client. It is **anonymous: there is no login, no account and no API key**, and it costs nothing.
- `cloudflared` opens an outbound connection from your PC to Cloudflare and is handed a random
  `https://<words>.trycloudflare.com` address that forwards back to the app. Because the connection is
  outbound, it works behind firewalls that block inbound traffic, and Cloudflare provides the HTTPS that
  phones need for camera access.
- The address changes every run and stops working when you press Ctrl+C. Quick tunnels have no uptime
  guarantee and are meant for testing and demos. A permanent address needs a free Cloudflare account and a
  domain (a "named tunnel").
- **Privacy:** only transport goes through Cloudflare. Camera frames and results are relayed over the tunnel
  between the phone and your PC, so Cloudflare's network carries them; the model and all processing stay on
  your PC. Don't point a camera at anything you wouldn't want to pass through a third party.
- **Security:** the link is the only secret, so don't post it publicly. Anyone who has it can open the
  detector, but a computer must still accept each phone. In this mode the app listens on `127.0.0.1` only,
  not on your LAN.

### Three ways to run it

| Command | Reach | Use it for |
|---|---|---|
| `uv run ods` | `http://localhost:8000` on this computer | The laptop's own camera |
| `uv run ods --share` | Public `https://...trycloudflare.com` link | Phone camera from anywhere, through firewalls |
| `uv run ods --https` | `https://<LAN-IP>:8000` with a self-signed certificate | Phone on the same Wi-Fi, when the PC accepts inbound connections |

Port 8000 can be changed with the `PORT` environment variable. In `--share` mode the app picks a free port
by itself if 8000 is taken, and you only ever need the public link.

## Run

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/). Run commands from the repo root
(`models.yaml`, `static/` and `weights/` are found relative to it). The first `uv sync` downloads PyTorch, which is large.

```bash
uv sync                       # creates .venv and installs everything from uv.lock
uv run ods                    # http://localhost:8000  (same as: uv run python -m ods.server)
uv run pytest                 # run the tests
```

**Phones:** browsers only allow camera access on `https://` or `localhost`. The easiest way to get a public
HTTPS link from your own PC is `uv run ods --share` (below). `uv run ods --https` instead serves a
self-signed certificate on your LAN (needs the phone to reach your PC directly, which some firewalls block).

## Share from your PC with one command

```bash
uv run ods --share
```

Starts the app and a free Cloudflare quick tunnel, then prints a public `https://<random>.trycloudflare.com`
link. The tunnel only makes an *outbound* connection, so it works even when the PC's firewall blocks inbound
traffic. The first run downloads the `cloudflared` client into `tools/` (or uses one already on your PATH).
The link changes on every run and stops working when you press Ctrl+C. Open the same link on the laptop and
the phone. Anyone who has the link can use the detector, but a computer still must accept a phone.

## Phone as the camera, results on the laptop

```
phone /phone ──request──▶ server ──popup──▶ laptop (Accept)
phone camera ──JPEG──▶ server ──YOLO with the laptop's classes──▶ frame + boxes ──▶ laptop
```

1. On the laptop open the site and tick **Accept phone cameras on this computer**. It shows a 6-digit code
   and stays listed while the page is open (the choice is remembered in the browser).
2. On the phone open the same site (phones are redirected to `/phone`; use `/?local` to detect on the phone
   itself). It lists the computers on the **same network** - the ones that have been waiting longest first -
   or you can type the laptop's code if they are not on the same network (e.g. phone on mobile data).
3. Tap a computer. The laptop shows an **Accept / Decline** popup (extra requests queue in order).
4. After Accept the phone streams its camera; the laptop picks the classes and sees the live boxes.
   Either side can stop at any time. One phone per computer at a time; others are told it is busy.

"Nearby" means "same public IP as seen by the server" (forwarded headers are honoured behind a proxy or
tunnel). Discovery is only a convenience: nothing streams until the laptop's user accepts. State is in memory,
so run a single server process. Protocol: see `ods/hub.py`.

## Hosting in the cloud (optional)

The repo ships a `Dockerfile` (CPU-only PyTorch, listens on `$PORT`), tested with `docker build`/`docker run`,
so any Docker host works (Cloud Run, Fly.io, a VPS). For most uses `uv run ods --share` (Cloudflare
tunnel, above) is enough and needs no cloud host. Run a single instance, because phone discovery state
is in memory. Free tiers with 512 MB RAM are too small for PyTorch.

Test the image locally with `docker build -t ods . && docker run -p 7860:7860 ods`.

## Choosing what can be detected – `models.yaml`

Each selectable class belongs to one model. The server runs only the models needed for the selected classes,
and passes the class IDs to YOLO so every other class is discarded. Add a new detectable thing by adding a
model (or classes) to `models.yaml`; no code changes.

- **person, car, truck, bus, bird, …** → `coco` model (`yolo11m.pt`, auto-downloaded). Swap for `yolo11l/x.pt`
  for more accuracy or `yolo11n/s.pt` for speed.
- **drone** → COCO has **no drone class**, so the `drone` model needs weights you provide at `weights/drone.pt`.
  Until then "drone" shows greyed out in the UI.

### Drone detection

Download a community drone model and verify it (run on a machine with internet access to Hugging Face):

```bash
uv run python scripts/get_model.py <hf_user>/<repo> --file best.pt      # saves weights/drone.pt, prints class names
```

Candidates to try (unverified - check licence, training data and test on your own camera):
[RohanSardar/yolo11s_drone_tracker](https://huggingface.co/RohanSardar/yolo11s_drone_tracker) (thermal footage only, AGPL-3.0),
[Tuzelkhan/drone-yolov8](https://huggingface.co/Tuzelkhan/drone-yolov8). Avoid VisDrone-trained models (they detect
things seen *from* drones, and are research-only) and "mock drone" models.

If you find none that works on your footage, train one on an open drone dataset:

```bash
uv run yolo detect train data=drone.yaml model=yolo11m.pt imgsz=960 epochs=100
cp runs/detect/train/weights/best.pt weights/drone.pt
```

If the model's class isn't literally `drone`, edit `classes: {drone: <name in model>}` in `models.yaml`.
Use a higher `imgsz` because drones are small in frame.

## Licensing note

Ultralytics YOLO11 is AGPL-3.0 (commercial licence available). If that doesn't fit, `Detector` takes a
`loader` callable, so an Apache/MIT model (YOLOX, RT-DETR via other runtimes, etc.) can be plugged in
without touching the server or UI.

## Layout

- `ods/registry.py` – parses `models.yaml`, maps class → model
- `ods/detector.py` – class-filtered inference
- `ods/server.py` – HTTP + WebSocket API, command-line modes (`--share`, `--https`)
- `ods/hub.py` – phone ↔ computer discovery, accept flow and frame relay
- `ods/share.py` – starts the Cloudflare quick tunnel and reads its public URL
- `static/index.html` – laptop UI: camera, class picker, overlay, phone requests
- `static/phone.html` – phone UI: nearby computers, camera streaming
- `scripts/get_model.py` – download and check a community model
- `Dockerfile` – container image for cloud hosting
- `tests/` – model filtering, pairing protocol, tunnel launcher
