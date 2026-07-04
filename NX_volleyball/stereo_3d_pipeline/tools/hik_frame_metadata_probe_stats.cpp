#include "hik_frame_metadata_probe_stats.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace hik_metadata_probe {

RunningStats::RunningStats()
    : min_v(std::numeric_limits<double>::infinity()),
      max_v(-std::numeric_limits<double>::infinity())
{
}

void RunningStats::add(double v) {
    ++n;
    min_v = std::min(min_v, v);
    max_v = std::max(max_v, v);
    const double delta = v - mean;
    mean += delta / static_cast<double>(n);
    const double delta2 = v - mean;
    m2 += delta * delta2;
}

double RunningStats::stddev() const {
    return n > 1 ? std::sqrt(m2 / static_cast<double>(n - 1)) : 0.0;
}

bool changed(uint32_t a, uint32_t b) {
    return a != b;
}

uint32_t counterStep(uint32_t prev, uint32_t cur) {
    return cur - prev;
}

void printIntervalStats(const char* label, const RunningStats& stats) {
    if (stats.n <= 0) {
        std::printf("  %s: no samples\n", label);
        return;
    }
    const double fps = stats.mean > 0.0 ? 1.0e9 / stats.mean : 0.0;
    std::printf("  %s: n=%d mean=%.1fns min=%.1f max=%.1f std=%.1f fps=%.3f\n",
                label, stats.n, stats.mean, stats.min_v, stats.max_v,
                stats.stddev(), fps);
}

void printMetaLine(int i, const FrameMeta& l, const FrameMeta& r) {
    if (!l.ok || !r.ok) {
        std::printf("[%04d] grab fail L(ok=%d ret=0x%X) R(ok=%d ret=0x%X)\n",
                    i, l.ok ? 1 : 0, l.ret, r.ok ? 1 : 0, r.ret);
        return;
    }
    const int64_t frame_delta = static_cast<int64_t>(l.frame_num) - static_cast<int64_t>(r.frame_num);
    const int64_t counter_delta = static_cast<int64_t>(l.frame_counter) - static_cast<int64_t>(r.frame_counter);
    const int64_t trigger_delta = static_cast<int64_t>(l.trigger_index) - static_cast<int64_t>(r.trigger_index);
    const int64_t ts_delta = static_cast<int64_t>(l.dev_ts) - static_cast<int64_t>(r.dev_ts);
    const int64_t host_delta = l.host_ts - r.host_ts;
    std::printf(
        "[%04d] L#%u R#%u d#=%ld  Lfc=%u Rfc=%u dfc=%ld  Ltr=%u Rtr=%u dtr=%ld  "
        "dDevTs=%ld dHostTs=%ld  Lsc/cyc/off=%u/%u/%u R=%u/%u/%u  chunk=%u/%u  "
        "first=%02X%02X%02X%02X/%02X%02X%02X%02X\n",
        i,
        l.frame_num, r.frame_num, static_cast<long>(frame_delta),
        l.frame_counter, r.frame_counter, static_cast<long>(counter_delta),
        l.trigger_index, r.trigger_index, static_cast<long>(trigger_delta),
        static_cast<long>(ts_delta), static_cast<long>(host_delta),
        l.second_count, l.cycle_count, l.cycle_offset,
        r.second_count, r.cycle_count, r.cycle_offset,
        l.unparsed_chunks, r.unparsed_chunks,
        l.first_bytes[0], l.first_bytes[1], l.first_bytes[2], l.first_bytes[3],
        r.first_bytes[0], r.first_bytes[1], r.first_bytes[2], r.first_bytes[3]);
}

}  // namespace hik_metadata_probe
