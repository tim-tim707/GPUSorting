# Minimal WebGPU demo

This small demo initializes WebGPU on a canvas and runs a simple clear-only render loop.

Files added:

- `index.html` — minimal page with a full-viewport `<canvas>` and overlay message.
- `main.js` — JavaScript module that requests an adapter/device, configures the `webgpu` context, handles resize, and starts a render loop.

How to run:

1. Serve the folder over a local HTTP server (browsers block many GPU APIs on `file://`).

   In PowerShell, from the project folder run:

```powershell
# using Python (if available)
python -m http.server 8000

# or, with Node.js installed:
npx http-server -p 8000
```

2. Open http://localhost:8000/ in a browser with WebGPU support (modern Chromium-based browsers like Chrome/Edge). If your browser doesn't advertise WebGPU, try an experimental build or enable the flag for WebGPU.

Notes / troubleshooting:
- If you see "WebGPU not supported in this browser.", try a newer Chrome/Edge or enable WebGPU flags.
- Open the devtools console to see errors if adapter/device requests fail.
