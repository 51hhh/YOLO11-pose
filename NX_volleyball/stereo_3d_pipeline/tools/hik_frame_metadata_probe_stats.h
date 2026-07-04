#pragma once

#include <cstdint>

namespace hik_metadata_probe {

struct FrameMeta {
    bool ok = false;
    int ret = 0;
    uint32_t frame_num = 0;
    uint64_t dev_ts = 0;
    int64_t host_ts = 0;
    uint32_t frame_counter = 0;
    uint32_t trigger_index = 0;
    uint32_t second_count = 0;
    uint32_t cycle_count = 0;
    uint32_t cycle_offset = 0;
    float gain = 0.0f;
    float exposure = 0.0f;
    uint32_t input = 0;
    uint32_t output = 0;
    uint32_t unparsed_chunks = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint64_t frame_len = 0;
    uint8_t first_bytes[16] = {};
};

struct RunningStats {
    int n = 0;
    double mean = 0.0;
    double m2 = 0.0;
    double min_v;
    double max_v;

    RunningStats();
    void add(double v);
    double stddev() const;
};

bool changed(uint32_t a, uint32_t b);
uint32_t counterStep(uint32_t prev, uint32_t cur);
void printIntervalStats(const char* label, const RunningStats& stats);
void printMetaLine(int i, const FrameMeta& l, const FrameMeta& r);

}  // namespace hik_metadata_probe
