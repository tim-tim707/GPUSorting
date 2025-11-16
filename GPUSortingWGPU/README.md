# WebGPU Radixsort

Adapted directly from (brush-sort)[https://github.com/ArthurBrussee/brush/tree/main/crates/brush-sort],
this sort can be used for back-to-front gaussian splatting sorting or any other sort that need to run on millions of elements as fast as possible

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

3. Check the results in the devtools console.

Notes / troubleshooting:
- If you see "WebGPU not supported in this browser.", try a newer Chrome/Edge or enable WebGPU flags.
- Open the devtools console to see errors if adapter/device requests fail.

## Performance considerations

I haven't profiled this sort with timestamp queries since they are not supported everywhere.
Sort from submit to CPU mapping can take 100-200ms but I suspect it is faster when no mapping is involved.
Also note that the first submit after loading the page can take up to 1-2 seconds.

There is room for improvement with new wgsl subgroups
