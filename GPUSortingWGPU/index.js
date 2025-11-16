// python -m http.server 8000
const overlay = document.getElementById("overlay");
const canvas = document.getElementById("gpuCanvas");
let adapter = null;
let device = null;

async function initWebGPU() {
    if (!("gpu" in navigator)) {
        overlay.textContent = "WebGPU not supported in this browser.";
        throw new Error("WebGPU not supported");
    }

    overlay.textContent = "Requesting GPU adapter…";
    adapter = await navigator.gpu.requestAdapter();

    if (!adapter) {
        overlay.textContent = "No suitable GPU adapter found.";
        throw new Error("No GPU adapter");
    }

    overlay.textContent = "Requesting GPU device…";
    device = await adapter.requestDevice({
        requiredFeatures: [],
    });
    overlay.textContent = "Configuring canvas for WebGPU…";

    const context = canvas.getContext("webgpu");
    const format = navigator.gpu.getPreferredCanvasFormat();

    function configureCanvas() {
        const dpr = Math.max(1, window.devicePixelRatio || 1);
        canvas.width = Math.floor(canvas.clientWidth * dpr);
        canvas.height = Math.floor(canvas.clientHeight * dpr);
        context.configure({
            device,
            format,
            alphaMode: "opaque",
        });
    }

    configureCanvas();
    new ResizeObserver(configureCanvas).observe(canvas);

    overlay.textContent = "WebGPU initialized — running.";

    window.runComputeAdd = runComputeAdd;
    const runBtn = document.getElementById("runComputeBtn");
    if (runBtn) {
        runBtn.addEventListener("click", async () => {
            overlay.textContent = "Compute running…";
            try {
                const resultArray = await runComputeAdd(2_000_000);

                overlay.textContent =
                    "Compute finished — result logged to console.";
            } catch (err) {
                console.error("Compute example failed", err);
                overlay.textContent =
                    "Compute example failed — see console for details.";
            }
        });
    }

    return { adapter, device, context };
}

window.addEventListener("DOMContentLoaded", () => {
    initWebGPU().catch((err) => {
        console.error(err);
        overlay.textContent = err.message || String(err);
    });
});

async function runComputeAdd(elementCount) {
    let stdout = "";
    window.__runComputeAddCpuStart = performance.now();
    overlay.textContent = `Preparing compute buffers (${elementCount} elements)…`;

    const bytesPerElement = 4;
    const bufferSize = elementCount * bytesPerElement;

    const storageBuffer = device.createBuffer({
        size: bufferSize,
        usage:
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_SRC |
            GPUBufferUsage.COPY_DST,
    });
    const storageBufferValues = device.createBuffer({
        size: bufferSize,
        usage:
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_SRC |
            GPUBufferUsage.COPY_DST,
    });

    overlay.textContent = "Uploading initial data to GPU…";
    const initArray = new Uint32Array(elementCount);
    for (let i = 0; i < elementCount; ++i) initArray[i] = elementCount - i + 1; // reverse sorted

    const initArrayValues = new Uint32Array(elementCount);
    for (let i = 0; i < elementCount; ++i) initArrayValues[i] = initArray[i] * 2;

    function printArray(array, begin, end) {
        return "[" + array.slice(begin, end).join(", ") + "]";
    }

    stdout += `Input keys:\n${printArray(initArray, 0, 300)}\n${printArray(initArray, elementCount - 300, elementCount)}\n`;
    stdout += `Input values:\n${printArray(initArrayValues, 0, 300)}\n${printArray(initArrayValues, elementCount - 300, elementCount)}\n`;

    device.queue.writeBuffer(storageBuffer, 0, initArray.buffer, 0, bufferSize);
    device.queue.writeBuffer(storageBufferValues, 0, initArrayValues.buffer, 0, bufferSize);

    let maxWorkgroupSize = (device.limits && device.limits.maxComputeWorkgroupSizeX) || 64;

    // Hardcoded in the shader for now
    maxWorkgroupSize = 256;

    const BIN_COUNT = 16;
    const BLOCK_SIZE = maxWorkgroupSize * 4; // ELEMENTS_PER_THREAD = 4
    const numWorkgroups = Math.ceil(elementCount / BLOCK_SIZE);
    const num_reduce_wgs = BIN_COUNT * Math.ceil(numWorkgroups / BLOCK_SIZE);

    const count_shadercode = /* wgsl */`
const OFFSET: u32 = 42;
const WG: u32 = 256;

const BITS_PER_PASS: u32 = 4;
const BIN_COUNT: u32 = 1u << BITS_PER_PASS;
const HISTOGRAM_SIZE: u32 = WG * BIN_COUNT;
const ELEMENTS_PER_THREAD: u32 = 4;

const BLOCK_SIZE = WG * ELEMENTS_PER_THREAD;

fn div_ceil(a: u32, b: u32) -> u32 {
    return (a + b - 1u) / b;
}

struct Uniforms {
    shift: u32,
}

@group(0) @binding(0) var<storage, read> config: Uniforms;
@group(0) @binding(1) var<storage, read> num_keys_arr: array<u32>;
@group(0) @binding(2) var<storage, read> src: array<u32>;
@group(0) @binding(3) var<storage, read_write> counts: array<u32>;

var<workgroup> histogram: array<atomic<u32>, BIN_COUNT>;

@compute
@workgroup_size(WG, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) gid: vec3<u32>,
) {
    let num_keys = num_keys_arr[0];

    let num_wgs = div_ceil(num_keys, BLOCK_SIZE);
    let group_id = gid.x;

    if group_id >= num_wgs {
        return;
    }

    if local_id.x < BIN_COUNT {
        atomicStore(&histogram[local_id.x], 0u);
    }
    workgroupBarrier();

    let wg_block_start = BLOCK_SIZE * group_id;
    var block_index = wg_block_start + local_id.x;
    let shift_bit = config.shift;
    var data_index = block_index;

    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        if data_index < num_keys {
            let local_key = (src[data_index] >> shift_bit) & 0xfu;
            atomicAdd(&histogram[local_key], 1u);
        }
        data_index += WG;
    }
    block_index += BLOCK_SIZE;
    workgroupBarrier();
    if local_id.x < BIN_COUNT {
        let num_wgs = div_ceil(num_keys, BLOCK_SIZE);
        counts[local_id.x * num_wgs + group_id] = atomicLoad(&histogram[local_id.x]);
    }
}`;
    const reduce_shadercode = /* wgsl */`
const OFFSET: u32 = 42;
const WG: u32 = 256;

const BITS_PER_PASS: u32 = 4;
const BIN_COUNT: u32 = 1u << BITS_PER_PASS;
const HISTOGRAM_SIZE: u32 = WG * BIN_COUNT;
const ELEMENTS_PER_THREAD: u32 = 4;

const BLOCK_SIZE = WG * ELEMENTS_PER_THREAD;

fn div_ceil(a: u32, b: u32) -> u32 {
    return (a + b - 1u) / b;
}

@group(0) @binding(0) var<storage, read> num_keys_arr: array<u32>;
@group(0) @binding(1) var<storage, read> counts: array<u32>;
@group(0) @binding(2) var<storage, read_write> reduced: array<u32>;

var<workgroup> sums: array<u32, WG>;

@compute
@workgroup_size(WG, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) gid: vec3<u32>,
) {
    let num_keys = num_keys_arr[0];
    let num_wgs = div_ceil(num_keys, BLOCK_SIZE);
    let num_reduce_wgs = BIN_COUNT * div_ceil(num_wgs, BLOCK_SIZE);

    let group_id = gid.x;

    if group_id >= num_reduce_wgs {
        return;
    }

    let num_reduce_wg_per_bin = num_reduce_wgs / BIN_COUNT;
    let bin_id = group_id / num_reduce_wg_per_bin;

    let bin_offset = bin_id * num_wgs;
    let base_index = (group_id % num_reduce_wg_per_bin) * BLOCK_SIZE;
    var sum = 0u;
    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let data_index = base_index + i * WG + local_id.x;
        if data_index < num_wgs {
            sum += counts[bin_offset + data_index];
        }
    }
    sums[local_id.x] = sum;
    for (var i = 0u; i < 8u; i++) {
        workgroupBarrier();
        if local_id.x < ((WG / 2u) >> i) {
            sum += sums[local_id.x + ((WG / 2u) >> i)];
            sums[local_id.x] = sum;
        }
    }
    if local_id.x == 0u {
        reduced[group_id] = sum;
    }
}`;
    const scan_shadercode = /* wgsl */`
const OFFSET: u32 = 42;
const WG: u32 = 256;

const BITS_PER_PASS: u32 = 4;
const BIN_COUNT: u32 = 1u << BITS_PER_PASS;
const HISTOGRAM_SIZE: u32 = WG * BIN_COUNT;
const ELEMENTS_PER_THREAD: u32 = 4;

const BLOCK_SIZE = WG * ELEMENTS_PER_THREAD;

fn div_ceil(a: u32, b: u32) -> u32 {
    return (a + b - 1u) / b;
}

@group(0) @binding(0) var<storage, read> num_keys_arr: array<u32>;
@group(0) @binding(1) var<storage, read_write> reduced: array<u32>;

var<workgroup> sums: array<u32, WG>;
var<workgroup> lds: array<array<u32, WG>, ELEMENTS_PER_THREAD>;

@compute
@workgroup_size(WG, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let num_keys = num_keys_arr[0];
    let num_wgs = div_ceil(num_keys, BLOCK_SIZE);
    let num_reduce_wgs = BIN_COUNT * div_ceil(num_wgs, BLOCK_SIZE);

    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let data_index = i * WG + local_id.x;
        let col = (i * WG + local_id.x) / ELEMENTS_PER_THREAD;
        let row = (i * WG + local_id.x) % ELEMENTS_PER_THREAD;
        lds[row][col] = reduced[data_index];
    }
    workgroupBarrier();
    var sum = 0u;
    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let tmp = lds[i][local_id.x];
        lds[i][local_id.x] = sum;
        sum += tmp;
    }
    // workgroup prefix sum
    sums[local_id.x] = sum;
    for (var i = 0u; i < 8u; i++) {
        workgroupBarrier();
        if local_id.x >= (1u << i) {
            sum += sums[local_id.x - (1u << i)];
        }
        workgroupBarrier();
        sums[local_id.x] = sum;
    }
    workgroupBarrier();
    sum = 0u;
    if local_id.x > 0u {
        sum = sums[local_id.x - 1u];
    }
    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        lds[i][local_id.x] += sum;
    }
    // lds now contains exclusive prefix sum
    workgroupBarrier();
    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let data_index = i * WG + local_id.x;
        let col = (i * WG + local_id.x) / ELEMENTS_PER_THREAD;
        let row = (i * WG + local_id.x) % ELEMENTS_PER_THREAD;
        if data_index < num_reduce_wgs {
            reduced[data_index] = lds[row][col];
        }
    }
}`;

    const scanAdd_shadercode = /* wgsl */`
const OFFSET: u32 = 42;
const WG: u32 = 256;

const BITS_PER_PASS: u32 = 4;
const BIN_COUNT: u32 = 1u << BITS_PER_PASS;
const HISTOGRAM_SIZE: u32 = WG * BIN_COUNT;
const ELEMENTS_PER_THREAD: u32 = 4;

const BLOCK_SIZE = WG * ELEMENTS_PER_THREAD;

fn div_ceil(a: u32, b: u32) -> u32 {
    return (a + b - 1u) / b;
}

@group(0) @binding(0) var<storage, read> num_keys_arr: array<u32>;
@group(0) @binding(1) var<storage, read> reduced: array<u32>;
@group(0) @binding(2) var<storage, read_write> counts: array<u32>;

var<workgroup> sums: array<u32, WG>;
var<workgroup> lds: array<array<u32, WG>, ELEMENTS_PER_THREAD>;

@compute
@workgroup_size(WG, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) gid: vec3<u32>,
) {
    let num_keys = num_keys_arr[0];
    let num_wgs = div_ceil(num_keys, BLOCK_SIZE);
    let num_reduce_wgs = BIN_COUNT * div_ceil(num_wgs, BLOCK_SIZE);

    let group_id = gid.x;

    if group_id >= num_reduce_wgs {
        return;
    }

    let num_reduce_wg_per_bin = num_reduce_wgs / BIN_COUNT;

    let bin_id = group_id / num_reduce_wg_per_bin;
    let bin_offset = bin_id * num_wgs;
    let base_index = (group_id % num_reduce_wg_per_bin) * ELEMENTS_PER_THREAD * WG;

    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let data_index = base_index + i * WG + local_id.x;
        let col = (i * WG + local_id.x) / ELEMENTS_PER_THREAD;
        let row = (i * WG + local_id.x) % ELEMENTS_PER_THREAD;
        // This is not gated, we let robustness do it for us
        lds[row][col] = counts[bin_offset + data_index];
    }
    workgroupBarrier();
    var sum = 0u;
    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let tmp = lds[i][local_id.x];
        lds[i][local_id.x] = sum;
        sum += tmp;
    }
    // workgroup prefix sum
    sums[local_id.x] = sum;
    for (var i = 0u; i < 8u; i++) {
        workgroupBarrier();
        if local_id.x >= (1u << i) {
            sum += sums[local_id.x - (1u << i)];
        }
        workgroupBarrier();
        sums[local_id.x] = sum;
    }
    workgroupBarrier();
    sum = reduced[group_id];
    if local_id.x > 0u {
        sum += sums[local_id.x - 1u];
    }
    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        lds[i][local_id.x] += sum;
    }
    // lds now contains exclusive prefix sum
    // Note: storing inclusive might be slightly cheaper here
    workgroupBarrier();
    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let data_index = base_index + i * WG + local_id.x;
        let col = (i * WG + local_id.x) / ELEMENTS_PER_THREAD;
        let row = (i * WG + local_id.x) % ELEMENTS_PER_THREAD;
        if data_index < num_wgs {
            counts[bin_offset + data_index] = lds[row][col];
        }
    }
}`;

    const scatter_shadercode = /* wgsl */`
const OFFSET: u32 = 42;
const WG: u32 = 256;

const BITS_PER_PASS: u32 = 4;
const BIN_COUNT: u32 = 1u << BITS_PER_PASS;
const HISTOGRAM_SIZE: u32 = WG * BIN_COUNT;
const ELEMENTS_PER_THREAD: u32 = 4;

const BLOCK_SIZE = WG * ELEMENTS_PER_THREAD;

fn div_ceil(a: u32, b: u32) -> u32 {
    return (a + b - 1u) / b;
}

struct Uniforms {
    shift: u32,
}

@group(0) @binding(0) var<storage, read> config: Uniforms;
@group(0) @binding(1) var<storage, read> num_keys_arr: array<u32>;
@group(0) @binding(2) var<storage, read> src: array<u32>;
@group(0) @binding(3) var<storage, read> values: array<u32>;
@group(0) @binding(4) var<storage, read> counts: array<u32>;
@group(0) @binding(5) var<storage, read_write> out: array<u32>;
@group(0) @binding(6) var<storage, read_write> out_values: array<u32>;

var<workgroup> lds_sums: array<u32, WG>;
var<workgroup> lds_scratch: array<u32, WG>;
var<workgroup> bin_offset_cache: array<u32, WG>;
var<workgroup> local_histogram: array<atomic<u32>, BIN_COUNT>;

@compute
@workgroup_size(WG, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) gid: vec3<u32>,
) {
    let num_keys = num_keys_arr[0];
    let num_wgs = div_ceil(num_keys, BLOCK_SIZE);

    let group_id = gid.x;

    if group_id >= num_wgs {
        return;
    }

    if local_id.x < BIN_COUNT {
        bin_offset_cache[local_id.x] = counts[local_id.x * num_wgs + group_id];
    }
    workgroupBarrier();
    let wg_block_start = BLOCK_SIZE * group_id;
    let block_index = wg_block_start + local_id.x;
    var data_index = block_index;
    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        if local_id.x < BIN_COUNT {
            atomicStore(&local_histogram[local_id.x], 0u);
        }
        var local_key = ~0u;
        var local_value = 0u;

        if data_index < num_keys {
            local_key = src[data_index];
            local_value = values[data_index];
        }

        for (var bit_shift = 0u; bit_shift < BITS_PER_PASS; bit_shift += 2u) {
            let key_index = (local_key >> config.shift) & 0xfu;
            let bit_key = (key_index >> bit_shift) & 3u;
            var packed_histogram = 1u << (bit_key * 8u);
            // workgroup prefix sum
            var sum = packed_histogram;
            lds_scratch[local_id.x] = sum;
            for (var i = 0u; i < 8u; i++) {
                workgroupBarrier();
                if local_id.x >= (1u << i) {
                    sum += lds_scratch[local_id.x - (1u << i)];
                }
                workgroupBarrier();
                lds_scratch[local_id.x] = sum;
            }
            workgroupBarrier();
            packed_histogram = lds_scratch[WG - 1u];
            packed_histogram = (packed_histogram << 8u) + (packed_histogram << 16u) + (packed_histogram << 24u);
            var local_sum = packed_histogram;
            if local_id.x > 0u {
                local_sum += lds_scratch[local_id.x - 1u];
            }
            let key_offset = (local_sum >> (bit_key * 8u)) & 0xffu;

            lds_sums[key_offset] = local_key;
            workgroupBarrier();
            local_key = lds_sums[local_id.x];
            workgroupBarrier();

            lds_sums[key_offset] = local_value;
            workgroupBarrier();
            local_value = lds_sums[local_id.x];
            workgroupBarrier();
        }
        let key_index = (local_key >> config.shift) & 0xfu;
        atomicAdd(&local_histogram[key_index], 1u);
        workgroupBarrier();
        var histogram_local_sum = 0u;
        if local_id.x < BIN_COUNT {
            histogram_local_sum = atomicLoad(&local_histogram[local_id.x]);
        }
        // workgroup prefix sum of histogram
        var histogram_prefix_sum = histogram_local_sum;
        if local_id.x < BIN_COUNT {
            lds_scratch[local_id.x] = histogram_prefix_sum;
        }
        for (var i = 0u; i < 4u; i++) {
            workgroupBarrier();
            if local_id.x >= (1u << i) && local_id.x < BIN_COUNT {
                histogram_prefix_sum += lds_scratch[local_id.x - (1u << i)];
            }
            workgroupBarrier();
            if local_id.x < BIN_COUNT {
                lds_scratch[local_id.x] = histogram_prefix_sum;
            }
        }
        let global_offset = bin_offset_cache[key_index];
        workgroupBarrier();
        var local_offset = local_id.x;
        if key_index > 0u {
            local_offset -= lds_scratch[key_index - 1u];
        }
        let total_offset = global_offset + local_offset;
        if total_offset < num_keys {
            out[total_offset] = local_key;
            out_values[total_offset] = local_value;
        }
        if local_id.x < BIN_COUNT {
            bin_offset_cache[local_id.x] += atomicLoad(&local_histogram[local_id.x]);
        }
        workgroupBarrier();
        data_index += WG;
    }
}`;

    const count_shaderModule = device.createShaderModule({ code: count_shadercode });
    const reduce_shaderModule = device.createShaderModule({ code: reduce_shadercode });
    const scan_shaderModule = device.createShaderModule({ code: scan_shadercode });
    const scanAdd_shaderModule = device.createShaderModule({ code: scanAdd_shadercode });
    const scatter_shaderModule = device.createShaderModule({ code: scatter_shadercode });

    const count_pipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module: count_shaderModule, entryPoint: "main" },
    });
    const reduce_pipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module: reduce_shaderModule, entryPoint: "main" },
    });
    const scan_pipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module: scan_shaderModule, entryPoint: "main" },
    });
    const scanAdd_pipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module: scanAdd_shaderModule, entryPoint: "main" },
    });
    const scatter_pipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module: scatter_shaderModule, entryPoint: "main" },
    });


    const n_sort = device.createBuffer({
        label: 'n_sort',
        size: 1 * Uint32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const elementCountArray = new Uint32Array([elementCount]);
    device.queue.writeBuffer(n_sort, 0, elementCountArray.buffer, 0, 4);

    let cur_keys = storageBuffer;
    let cur_vals = storageBufferValues;

    overlay.textContent = "Dispatching compute shader…";

    const readBuffer = device.createBuffer({ size: bufferSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const readBufferKeys = device.createBuffer({ size: bufferSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const readBufferValues = device.createBuffer({ size: bufferSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const commandEncoder = device.createCommandEncoder({ label: 'commandEncoder' });

    const count_buf = device.createBuffer({ label: 'count_buf', size: numWorkgroups * BIN_COUNT * Uint32Array.BYTES_PER_ELEMENT, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const reduced_buf = device.createBuffer({ label: 'reduced_buf', size: BLOCK_SIZE * Uint32Array.BYTES_PER_ELEMENT, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

    const reduce_bindGroup = device.createBindGroup({
        layout: reduce_pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: n_sort } },
            { binding: 1, resource: { buffer: count_buf } },
            { binding: 2, resource: { buffer: reduced_buf } },
        ],
    });
    const scan_bindGroup = device.createBindGroup({
        layout: scan_pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: n_sort } },
            { binding: 1, resource: { buffer: reduced_buf } },
        ],
    });
    const scanAdd_bindGroup = device.createBindGroup({
        layout: scanAdd_pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: n_sort } },
            { binding: 1, resource: { buffer: reduced_buf } },
            { binding: 2, resource: { buffer: count_buf } },
        ],
    });

    let output_keys = device.createBuffer({ label: 'output_keys', size: elementCount * Uint32Array.BYTES_PER_ELEMENT, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    let output_values = device.createBuffer({ label: 'output_values', size: elementCount * Uint32Array.BYTES_PER_ELEMENT, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });

    const shifts = new Uint32Array([0]);

    for (let passIndex = 0; passIndex < 8; passIndex++) {
        // stdout += `pass ${passIndex} shift ${passIndex * 4}\n`;

        const passEven = passIndex % 2 === 0;
        const uniforms_buffer = device.createBuffer({ label: 'uniforms_buffer', size: 1 * Uint32Array.BYTES_PER_ELEMENT, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

        commandEncoder.clearBuffer(count_buf);
        commandEncoder.clearBuffer(reduced_buf);

        const pass = commandEncoder.beginComputePass({ label: 'computePass' });
        shifts[0] = passIndex * 4;
        device.queue.writeBuffer(uniforms_buffer, 0, shifts.buffer, 0, 4);

        const count_bindGroup = device.createBindGroup({
            layout: count_pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: uniforms_buffer } },
                { binding: 1, resource: { buffer: n_sort } },
                { binding: 2, resource: { buffer: passEven ? cur_keys : output_keys } },
                { binding: 3, resource: { buffer: count_buf } },
            ],
        });
        pass.setPipeline(count_pipeline);
        pass.setBindGroup(0, count_bindGroup);
        pass.dispatchWorkgroups(numWorkgroups);

        pass.setPipeline(reduce_pipeline);
        pass.setBindGroup(0, reduce_bindGroup);
        pass.dispatchWorkgroups(num_reduce_wgs);

        pass.setPipeline(scan_pipeline);
        pass.setBindGroup(0, scan_bindGroup);
        pass.dispatchWorkgroups(1, 1, 1);

        pass.setPipeline(scanAdd_pipeline);
        pass.setBindGroup(0, scanAdd_bindGroup);
        pass.dispatchWorkgroups(num_reduce_wgs);

        const scatter_bindGroup = device.createBindGroup({
            layout: scatter_pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: uniforms_buffer } },
                { binding: 1, resource: { buffer: n_sort } },
                { binding: 2, resource: { buffer: passEven ? cur_keys : output_keys } },
                { binding: 3, resource: { buffer: passEven ? cur_vals : output_values } },
                { binding: 4, resource: { buffer: count_buf } },
                { binding: 5, resource: { buffer: passEven ? output_keys : cur_keys } },
                { binding: 6, resource: { buffer: passEven ? output_values : cur_vals } },
            ],
        });

        pass.setPipeline(scatter_pipeline);
        pass.setBindGroup(0, scatter_bindGroup);
        pass.dispatchWorkgroups(numWorkgroups);

        pass.end();
    }

    commandEncoder.copyBufferToBuffer(storageBuffer, 0, readBuffer, 0, bufferSize);
    commandEncoder.copyBufferToBuffer(output_keys, 0, readBufferKeys, 0, bufferSize);
    commandEncoder.copyBufferToBuffer(output_values, 0, readBufferValues, 0, bufferSize);

    const cpuSubmitTime = performance.now();
    device.queue.submit([commandEncoder.finish()]);

    overlay.textContent = "Waiting for GPU and reading back results…";
    // Wait for readbacks
    await Promise.all([
        readBuffer.mapAsync(GPUMapMode.READ),
        readBufferKeys.mapAsync(GPUMapMode.READ),
        readBufferValues.mapAsync(GPUMapMode.READ),
    ]);
    const cpuMapTime = performance.now();
    const cpuSubmitToMapMs = cpuMapTime - cpuSubmitTime;
    console.log(`CPU time (submit -> mapAsync complete): ${cpuSubmitToMapMs.toFixed(3)} ms`);

    const mapped = readBuffer.getMappedRange();
    const resultArray = new Uint32Array(mapped.slice(0));
    const fullResultArray = new Uint32Array(resultArray);
    readBuffer.unmap();

    const mappedKeys = readBufferKeys.getMappedRange();
    const resultKeysArray = new Uint32Array(mappedKeys.slice(0));
    const fullResultKeysArray = new Uint32Array(resultKeysArray);
    readBufferKeys.unmap();

    const mappedValues = readBufferValues.getMappedRange();
    const resultValuesArray = new Uint32Array(mappedValues.slice(0));
    const fullResultValuesArray = new Uint32Array(resultValuesArray);
    readBufferValues.unmap();

    const cpuEnd = performance.now();
    const cpuTotalMs =
        cpuEnd - (window.__runComputeAddCpuStart || cpuSubmitTime);
    console.log(`CPU total time (function): ${cpuTotalMs.toFixed(3)} ms`);

    stdout += `Final keys:\n${printArray(fullResultKeysArray, 0, 300)}\n${printArray(fullResultKeysArray, elementCount - 300, elementCount)}\n`;
    stdout += `Final values:\n${printArray(fullResultValuesArray, 0, 300)}\n${printArray(fullResultValuesArray, elementCount - 300, elementCount)}\n`;

    console.log(stdout);

    return fullResultArray;
}
