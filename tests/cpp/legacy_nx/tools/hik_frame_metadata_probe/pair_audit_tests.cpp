#include "tools/hik_frame_metadata_probe/pair_audit.h"

#include <cstdint>
#include <iostream>
#include <limits>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAILED: " << message << '\n';
    }
}

hik_metadata_probe::FrameMeta metadata(
    std::uint32_t frame_number,
    std::uint32_t frame_counter,
    std::uint32_t trigger_index,
    std::uint64_t timestamp) {
    hik_metadata_probe::FrameMeta value;
    value.ok = true;
    value.frame_num = frame_number;
    value.frame_counter = frame_counter;
    value.trigger_index = trigger_index;
    value.dev_ts = timestamp;
    value.host_ts = static_cast<std::int64_t>(timestamp + 1000);
    return value;
}

void recordContinuousPair(
    hik_metadata_probe::PairAudit& audit,
    int sample_index,
    std::uint32_t identity) {
    const auto left = metadata(identity, identity, identity, identity * 10U);
    const auto right = metadata(identity, identity, identity, identity * 10U + 2U);
    check(audit.recordPair(sample_index, left, right),
          "valid stereo metadata must be accepted by the audit");
}

void testContinuousCountersPass() {
    hik_metadata_probe::PairAudit audit;
    recordContinuousPair(audit, 30, 100);
    recordContinuousPair(audit, 31, 101);
    check(audit.okPairs() == 2 && audit.failedPairs() == 0,
          "continuous pairs must be counted without failures");
    check(audit.passed(false),
          "equal, populated and continuous frame counters must pass");
}

void testPwmEvidenceIsRequiredWhenEnabled() {
    hik_metadata_probe::PairAudit audit;
    recordContinuousPair(audit, 30, 10);
    recordContinuousPair(audit, 31, 11);
    check(!audit.passed(true),
          "PWM mode must not pass without one mapping result per valid pair");
    audit.recordPwmMapping(true);
    audit.recordPwmMapping(true);
    check(audit.passed(true),
          "complete matching GPIO evidence must pass");

    hik_metadata_probe::PairAudit mismatch;
    recordContinuousPair(mismatch, 30, 20);
    recordContinuousPair(mismatch, 31, 21);
    mismatch.recordPwmMapping(true);
    mismatch.recordPwmMapping(false);
    check(!mismatch.passed(true),
          "one missing GPIO sequence mapping must fail the probe");
}

void testIdentityFailuresAreRejected() {
    hik_metadata_probe::PairAudit unequal;
    check(unequal.recordPair(
              30,
              metadata(1, 10, 20, 100),
              metadata(1, 11, 20, 102)),
          "readable but unequal first pair still contributes diagnostics");
    check(unequal.recordPair(
              31,
              metadata(2, 11, 21, 110),
              metadata(2, 12, 21, 112)),
          "second unequal pair still contributes diagnostics");
    check(!unequal.passed(false),
          "non-zero left/right FrameCounter offset must fail");

    hik_metadata_probe::PairAudit unpopulated;
    check(unpopulated.recordPair(
              30,
              metadata(1, 0, 0, 100),
              metadata(1, 0, 0, 102)),
          "zero-valued payload remains readable evidence");
    check(unpopulated.recordPair(
              31,
              metadata(2, 0, 0, 110),
              metadata(2, 0, 0, 112)),
          "second zero-valued payload remains readable evidence");
    check(!unpopulated.passed(false),
          "an unpopulated FrameCounter must never pass");

    hik_metadata_probe::PairAudit failed_grab;
    hik_metadata_probe::FrameMeta failed;
    failed.ret = -1;
    check(!failed_grab.recordPair(30, failed, failed),
          "failed SDK reads must not become valid pairs");
    check(failed_grab.failedPairs() == 1 && !failed_grab.passed(false),
          "failed SDK reads must make the final verdict fail");
}

void testUnsignedWrapIsContinuous() {
    hik_metadata_probe::PairAudit audit;
    const auto maximum = std::numeric_limits<std::uint32_t>::max();
    recordContinuousPair(audit, 30, maximum);
    recordContinuousPair(audit, 31, 0);
    check(audit.passed(false),
          "natural uint32 counter wrap must be treated as a +1 step");
}

}  // namespace

int main() {
    testContinuousCountersPass();
    testPwmEvidenceIsRequiredWhenEnabled();
    testIdentityFailuresAreRejected();
    testUnsignedWrapIsContinuous();
    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "hik_frame_metadata_probe pair audit tests passed\n";
    return 0;
}
