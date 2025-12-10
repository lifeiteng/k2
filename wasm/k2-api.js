
/**
 * k2-api.js
 * High-level JavaScript Wrapper for K2 WASM Module
 */

const createK2Module = require('../../build-wasm/k2-wasm.js'); // Relative to k2/wasm/

class K2Fsa {
    constructor() {
        this.module = null;
        this.readyPromise = this.init();
    }

    async init() {
        this.module = await createK2Module();
        return this.module;
    }

    async ready() {
        return this.readyPromise;
    }

    /**
     * Create an FsaVec in WASM memory from JS data.
     * @param {Object} fsaData - { row_splits1: Int32Array, row_splits2: Int32Array, arcs: { src: [], dest: [], label: [], score: [] } }
     */
    createFsaVec(fsaData) {
        if (!this.module) throw new Error("K2 Module not initialized");
        const m = this.module;


        const rs1 = new Int32Array(fsaData.row_splits1);
        const rs2 = new Int32Array(fsaData.row_splits2);

        const numArcs = fsaData.arcs.src.length;
        const src = new Int32Array(fsaData.arcs.src);
        const dest = new Int32Array(fsaData.arcs.dest);
        const label = new Int32Array(fsaData.arcs.label);
        const score = new Float32Array(fsaData.arcs.score);

        let heap32 = m.HEAP32;
        let heapF32 = m.HEAPF32;
        if (!heap32) {
            const buffer = m.buffer || (m.wasmMemory ? m.wasmMemory.buffer : null);
            if (buffer) {
                heap32 = new Int32Array(buffer);
                heapF32 = new Float32Array(buffer);
            } else {
                throw new Error("Cannot find memory buffer in Module");
            }
        }

        // Allocate memory
        const rs1Ptr = m._malloc(rs1.byteLength);
        const rs2Ptr = m._malloc(rs2.byteLength);
        const srcPtr = m._malloc(src.byteLength);
        const destPtr = m._malloc(dest.byteLength);
        const labelPtr = m._malloc(label.byteLength);
        const scorePtr = m._malloc(score.byteLength);

        // Copy data
        // CAUTION: Buffer might detach on malloc if growth happens?
        // With ALLOW_MEMORY_GROWTH, views can be invalidated.
        // We should re-get view after mallocs just in case?
        // But malloc happens first.
        // Let's re-get buffer view right before set.

        const updateViews = () => {
            if (m.HEAP32) {
                heap32 = m.HEAP32;
                heapF32 = m.HEAPF32;
            } else {
                const buffer = m.buffer || (m.wasmMemory ? m.wasmMemory.buffer : null);
                if (buffer) {
                    heap32 = new Int32Array(buffer);
                    heapF32 = new Float32Array(buffer);
                }
            }
        }
        updateViews();

        heap32.set(rs1, rs1Ptr >> 2);
        heap32.set(rs2, rs2Ptr >> 2);
        heap32.set(src, srcPtr >> 2);
        heap32.set(dest, destPtr >> 2);
        heap32.set(label, labelPtr >> 2);
        heapF32.set(score, scorePtr >> 2);

        // Create FsaVec
        // explicit number of fsas? inferred from rs1 size - 1
        const numFsas = rs1.length - 1;

        const fsaVec = m.CreateFsaVec(
            numFsas,
            rs1Ptr, rs1.length,
            rs2Ptr, rs2.length,
            srcPtr, destPtr, labelPtr, scorePtr,
            numArcs
        );

        // Free temporary memory
        // (Note: FsaVec copies data internally in our C++ implementation)
        m._free(rs1Ptr);
        m._free(rs2Ptr);
        m._free(srcPtr);
        m._free(destPtr);
        m._free(labelPtr);
        m._free(scorePtr);

        return fsaVec; // SharedPtr returned by Embind
    }

    /**
     * Helper to create FsaVec from string (PyK2 format).
     * @param {string} str
     */
    createFsaVecFromStr(str) {
        // Parse string line by line
        const lines = str.trim().split('\n');
        const arcs = [];
        let finalState = -1;

        for (let line of lines) {
            line = line.trim();
            if (!line) continue;
            const parts = line.split(/\s+/);
            if (parts.length === 1) {
                // Final state line
                finalState = parseInt(parts[0]);
            } else if (parts.length >= 3) {
                // Arc line
                // format: src dest label [aux_label] [score]
                const src = parseInt(parts[0]);
                const dest = parseInt(parts[1]);
                const label = parseInt(parts[2]);

                let score = 0.0;
                // Heuristic: if 5 parts, last is likely score.
                // If 4 parts, could be score or aux.
                // Given the user case: src dest label aux score (5 parts)
                if (parts.length === 5) {
                    score = parseFloat(parts[4]);
                } else if (parts.length === 4) {
                    // src dest label score
                    score = parseFloat(parts[3]);
                }

                arcs.push({ src, dest, label, score });
            }
        }

        // Sort arcs by src state (K2 usually expects sorted property, but raw construction manages)
        arcs.sort((a, b) => a.src - b.src);

        // Compute num states
        let maxState = finalState;
        for (const arc of arcs) {
            if (arc.src > maxState) maxState = arc.src;
            if (arc.dest > maxState) maxState = arc.dest;
        }
        if (maxState === -1 && arcs.length > 0) {
            // Fallback if no final line
            maxState = arcs[arcs.length - 1].src; // Rough guess
        }
        const numStates = maxState + 1;

        // Build row_splits1 (Fsa -> State)
        // Single FSA -> [0, numStates]
        const rs1 = new Int32Array([0, numStates]);

        // Build row_splits2 (State -> Arc)
        const rs2 = new Int32Array(numStates + 1);
        const arcCounts = new Int32Array(numStates);
        for (const arc of arcs) {
            if (arc.src < numStates) arcCounts[arc.src]++;
        }
        let cumsum = 0;
        for (let i = 0; i < numStates; i++) {
            rs2[i] = cumsum;
            cumsum += arcCounts[i];
        }
        rs2[numStates] = cumsum;

        // Flatten arcs
        const srcArr = new Int32Array(arcs.length);
        const destArr = new Int32Array(arcs.length);
        const labelArr = new Int32Array(arcs.length);
        const scoreArr = new Float32Array(arcs.length);

        for (let i = 0; i < arcs.length; i++) {
            srcArr[i] = arcs[i].src;
            destArr[i] = arcs[i].dest;
            labelArr[i] = arcs[i].label;
            scoreArr[i] = arcs[i].score;
        }

        return this.createFsaVec({
            row_splits1: rs1,
            row_splits2: rs2,
            arcs: {
                src: srcArr,
                dest: destArr,
                label: labelArr,
                score: scoreArr
            }
        });
    }

    /**
     * Arc Sort an FsaVec in-place.
     * @param {std::shared_ptr<FsaVec>} fsaVec
     */
    arcSort(fsaVec) {
        if (!this.module) throw new Error("K2 Module not initialized");
        this.module.ArcSort(fsaVec);
    }

    /**
     * Get Shortest Path of an FsaVec.
     * Returns a JS object representing the linear FsaVec { src: [], dest: [], ... }.
     * @param {std::shared_ptr<FsaVec>} fsaVec
     */
    shortestPath(fsaVec) {
        if (!this.module) throw new Error("K2 Module not initialized");
        return this.module.GetShortestPath(fsaVec);
    }

    /**
     * Get Lattice from NNet Output and Decoding Graph
     * @param {Object} nnetOutput - ONNX Runtime Tensor (or compatible object with data and dims)
     * @param {std::shared_ptr<FsaVec>} decodingGraph - Graph created via createFsaVec
     * @param {Object} config - { searchBeam, outputBeam, minActive, maxActive }
     */
    getLattice(nnetOutput, decodingGraph, config = {}) {
        if (!this.module) throw new Error("K2 Module not initialized");
        const m = this.module;

        const {
            searchBeam = 20.0,
            outputBeam = 20.0,
            minActive = 200,
            maxActive = 10000
        } = config;

        // nnetOutput assumed to be [batch, frames, classes] (DenseFsaVec)
        // For now support batch=1 or handle batching?
        // nnetOutput.dims = [1, T, C] or [T, C]

        let dims = nnetOutput.dims;
        let data = nnetOutput.data; // Float32Array usually

        let numRows, numCols;
        if (dims.length === 3) {
            numRows = dims[1]; // T
            numCols = dims[2]; // C
            // Flatten if needed? ort.Tensor data is already flat TypedArray
        } else if (dims.length === 2) {
            numRows = dims[0];
            numCols = dims[1];
        } else {
            throw new Error("Invalid nnetOutput dimensions");
        }

        const scoresSize = numRows * numCols * 4; // float32 bytes
        const scoresPtr = m._malloc(scoresSize);
        m.HEAPF32.set(data, scoresPtr >> 2);

        // Call Intersect
        let result;
        try {
            result = m.Intersect(
                decodingGraph,
                scoresPtr, numRows, numCols,
                searchBeam, outputBeam,
                minActive, maxActive
            );
        } catch (e) {
            console.error("Intersect failed with raw error:", e);
            if (typeof e === 'number' && m.getExceptionMessage) {
                const msg = m.getExceptionMessage(e);
                console.error("C++ Exception:", msg);
            }
            m._free(scoresPtr);
            throw new Error("Intersect Failed");
        }

        m._free(scoresPtr);

        return result; // contains { src, dest, label, score, row_splits1, row_splits2 }
    }
}

module.exports = K2Fsa;
