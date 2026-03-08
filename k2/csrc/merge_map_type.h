// Copyright 2025 LattifAI (authors: Feiteng Li)
//
// Use of this source code is governed by an Apache-2.0 license.
//
// merge_map values encode (src_idx * num_srcs + src) which can overflow
// uint32_t for long audio with many arcs.  Using uint64_t avoids the overflow.

#pragma once
#include <cstdint>

namespace k2 {
using merge_map_t = uint64_t;
}  // namespace k2
