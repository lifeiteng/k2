
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <vector>
#include <memory>
#include <iostream>

#include "k2/csrc/intersect_dense_pruned.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/host/dense_fsa.h"
#include "k2/csrc/fsa_utils.h"
#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/array.h"
#include "k2/csrc/context.h"

using namespace emscripten;
using namespace k2;

// --- Helper Functions to Create FsaVec and DenseFsaVec from memory views ---

// Helper to create Array1 from raw pointer
template <typename T>
Array1<T> CreateArray1(ContextPtr c, uintptr_t ptr, int32_t size) {
    // Note: This copies data. Zero-copy is harder because Array1 expects ownership or specific allocator.
    // For simpler implementation, we copy.
    Array1<T> ans(c, size);
    T* data = ans.Data();
    const T* src = reinterpret_cast<const T*>(ptr);
    std::copy(src, src + size, data);
    return ans;
}

// Function to create FsaVec from flat arrays
std::shared_ptr<FsaVec> CreateFsaVecWasm(
    int32_t num_fsas,
    uintptr_t row_splits1_ptr, int32_t row_splits1_size, // fsa -> state
    uintptr_t row_splits2_ptr, int32_t row_splits2_size, // state -> arc
    uintptr_t arcs_src_ptr, uintptr_t arcs_dest_ptr, uintptr_t arcs_label_ptr, uintptr_t arcs_score_ptr, // Arcs struct of arrays
    int32_t num_arcs
) {
    ContextPtr c = GetCpuContext();

    // 1. Create RaggedShape
    // row_splits1
    Array1<int32_t> rs1 = CreateArray1<int32_t>(c, row_splits1_ptr, row_splits1_size);
    // row_splits2
    Array1<int32_t> rs2 = CreateArray1<int32_t>(c, row_splits2_ptr, row_splits2_size);

    // We assume row_splits are valid and on CPU.
    RaggedShape shape = RaggedShape3(&rs1, nullptr, -1, &rs2, nullptr, -1);

    // 2. Create Arcs
    Array1<Arc> arcs(c, num_arcs);
    Arc* arcs_data = arcs.Data();

    const int32_t* src = reinterpret_cast<const int32_t*>(arcs_src_ptr);
    const int32_t* dest = reinterpret_cast<const int32_t*>(arcs_dest_ptr);
    const int32_t* label = reinterpret_cast<const int32_t*>(arcs_label_ptr);
    const float* score = reinterpret_cast<const float*>(arcs_score_ptr);

    for(int i=0; i<num_arcs; ++i) {
        arcs_data[i].src_state = src[i];
        arcs_data[i].dest_state = dest[i];
        arcs_data[i].label = label[i];
        arcs_data[i].score = score[i];
    }

    return std::make_shared<FsaVec>(shape, arcs);
}

// Helper to convert FsaVec to simple JS object
val FsaVecToJS(const FsaVec& fsa_vec) {
    val res = val::object();
    int32_t num_arcs = fsa_vec.NumElements();

    // Safety check for empty FSA (uninitialized values)
    if (!fsa_vec.values.IsValid()) {
        res.set("src", val::array());
        res.set("dest", val::array());
        res.set("label", val::array());
        res.set("score", val::array());
        res.set("row_splits1", val::array());
        res.set("row_splits2", val::array());
        return res;
    }

    const Arc* arcs_data = fsa_vec.values.Data();

    std::vector<int32_t> src(num_arcs), dest(num_arcs), label(num_arcs);
    std::vector<float> score(num_arcs);

    for(int i=0; i<num_arcs; ++i) {
        src[i] = arcs_data[i].src_state;
        dest[i] = arcs_data[i].dest_state;
        label[i] = arcs_data[i].label;
        score[i] = arcs_data[i].score;
    }

    res.set("src", val::array(src));
    res.set("dest", val::array(dest));
    res.set("label", val::array(label));
    res.set("score", val::array(score));

    const int32_t* rs1_data = fsa_vec.shape.RowSplits(1).Data();
    std::vector<int32_t> rs1(rs1_data, rs1_data + fsa_vec.shape.RowSplits(1).Dim());
    res.set("row_splits1", val::array(rs1));

    const int32_t* rs2_data = fsa_vec.shape.RowSplits(2).Data();
    std::vector<int32_t> rs2(rs2_data, rs2_data + fsa_vec.shape.RowSplits(2).Dim());
    res.set("row_splits2", val::array(rs2));

    return res;
}

// Function to run Intersection
// Returns a JS Object with results
val IntersectWasm(std::shared_ptr<FsaVec> decoding_graph,
                   uintptr_t dense_fsa_scores_ptr, int32_t num_dense_rows, int32_t num_dense_cols,
                   float search_beam, float output_beam,
                   int32_t min_active, int32_t max_active) {
    ContextPtr c = GetCpuContext();

    // Construct DenseFsaVec
    // We assume batch size 1 for now or infer from decoding_graph?
    // K2 usually assumes batch size match.
    // If decoding_graph has Dim0 = 1, it broadcasts.
    // Let's assume DenseFsaVec has only 1 sequence i.e. 1 chunk of audio.
    // So shape is [1][num_dense_rows].

    // Output lattice placeholder
    FsaVec out_fsa;
    Array1<int32_t> arc_map_a;
    Array1<int32_t> arc_map_b;

    // DenseFsaVec shape
    // We add 1 extra row for the "final frame" (all -inf usually, unless specific final transition handling is needed)
    // K2 Internal logic expects DenseFsaVec to have this extra frame.
    int32_t actual_rows = num_dense_rows + 1;
    Array1<int32_t> shape_rs1(c, 2);
    shape_rs1.Data()[0] = 0;
    shape_rs1.Data()[1] = actual_rows;
    RaggedShape dense_shape = RaggedShape2(&shape_rs1, nullptr, -1);

    // Scores: [actual_rows, num_dense_cols + 1]
    // We add 1 column padding at index 0 because k2 IntersectDensePruned shifts label by 1 (Label k -> Col k+1).
    // So Column 0 is for Label -1. Columns 1..N are for Label 0..N-1.
    int32_t actual_cols = num_dense_cols + 1;
    Array2<float> scores(c, actual_rows, actual_cols);
    float* scores_data = scores.Data();
    const float* input_scores = reinterpret_cast<const float*>(dense_fsa_scores_ptr);

    // Initialize with -inf
    std::fill(scores_data, scores_data + actual_rows * actual_cols, -std::numeric_limits<float>::infinity());

    // Copy input scores to columns 1..N
    // Input is row-major: [num_dense_rows, num_dense_cols]
    for(int i = 0; i < num_dense_rows; ++i) {
        float* row_ptr = scores_data + i * actual_cols;
        const float* input_row = input_scores + i * num_dense_cols;
        std::copy(input_row, input_row + num_dense_cols, row_ptr + 1);

        // Set Col 0 (Label -1) to 0.0f to allow final transitions?
        // Actually usually arcs with -1 label are for final transitions.
        row_ptr[0] = 0.0f;
    }

    // Last row (actual_rows - 1) is already -inf.
    // We set Col 0 of last row to 0.0f to allow "transition" logic if needed.
    // (Though last row is usually mostly placeholder, but arc_map logic looks at it).
    float* last_row = scores_data + num_dense_rows * actual_cols;
    last_row[0] = 0.0f;

    DenseFsaVec dense_fsa_vec(dense_shape, scores);

    // Run IntersectDensePruned
    // Usage of allow_partial=true matches typical "aligner" usage where we might not reach end perfectly,
    // but the test data generator might use default (False).
    // Let's stick with True for robustness, but if mismatch persists, we might switch.
    IntersectDensePruned(
        *decoding_graph,
        dense_fsa_vec,
        search_beam,
        output_beam,
        min_active,
        max_active,
        true, // allow_partial
        &out_fsa,
        &arc_map_a,
        &arc_map_b
    );

    return FsaVecToJS(out_fsa);
}

// Wrapper for ArcSort
void ArcSortWrapper(std::shared_ptr<FsaVec> fsa_vec) {
    if (fsa_vec) {
        ArcSort(fsa_vec.get());
    }
}

// Wrapper for ShortestPath
val GetShortestPath(std::shared_ptr<FsaVec> fsa_vec) {
     if (!fsa_vec || fsa_vec->NumElements() == 0) return FsaVecToJS(*fsa_vec);

     ContextPtr c = fsa_vec->Context();

     // ShortestPath pipeline
     Ragged<int32_t> state_batches = GetStateBatches(*fsa_vec, true);
     Array1<int32_t> dest_states = GetDestStates(*fsa_vec, true);
     Ragged<int32_t> incoming_arcs = GetIncomingArcs(*fsa_vec, dest_states);
     Ragged<int32_t> entering_arc_batches = GetEnteringArcIndexBatches(*fsa_vec, incoming_arcs, state_batches);

     bool log_semiring = false; // Viterbi
     Array1<int32_t> entering_arcs;
     // Using double for accumulation logic similar to generic implementation
     GetForwardScores<double>(*fsa_vec, state_batches, entering_arc_batches, log_semiring, &entering_arcs);

     Ragged<int32_t> best_path_arc_indexes = ShortestPath(*fsa_vec, entering_arcs);
     FsaVec ans = FsaVecFromArcIndexes(*fsa_vec, best_path_arc_indexes);

     return FsaVecToJS(ans);
}

EMSCRIPTEN_BINDINGS(k2_module) {
    class_<FsaVec>("FsaVec")
        .smart_ptr<std::shared_ptr<FsaVec>>("std::shared_ptr<FsaVec>");

    function("CreateFsaVec", &CreateFsaVecWasm);
    function("Intersect", &IntersectWasm);
    function("ArcSort", &ArcSortWrapper);
    function("GetShortestPath", &GetShortestPath);
}
