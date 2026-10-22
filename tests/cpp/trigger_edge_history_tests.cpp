#include "calibration/pwm_trigger.h"

#include <cstdlib>
#include <iostream>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAILED: " << message << '\n';
    }
}

stereo3d::PWMTriggerEdgeEvent event(std::uint64_t sequence) {
    return {
        sequence,
        1000 + sequence * 100,
        1010 + sequence * 100,
        2000 + sequence * 100,
        20,
    };
}

void testLookupIsStableAndBounded() {
    stereo3d::PWMTriggerEventHistory history(3);
    history.push(event(1));
    history.push(event(2));
    history.push(event(3));
    stereo3d::PWMTriggerEdgeEvent output;
    check(history.find(2, output), "retained trigger must be found");
    check(output.edge_realtime_ns == 2200, "trigger fields must remain unchanged");
    history.push(event(4));
    check(!history.find(1, output), "oldest trigger must leave bounded history");
    check(history.find(4, output), "latest trigger must be found");
    check(history.latestSequence() == 4, "latest sequence must be auditable");
    check(history.size() == 3, "history must honor its capacity");
}

void testNonMonotonicEventsAreRejected() {
    stereo3d::PWMTriggerEventHistory history;
    history.push(event(2));
    history.push(event(2));
    history.push(event(1));
    check(history.size() == 1, "duplicate and reversed events must be ignored");
    history.clear();
    check(history.size() == 0, "clear must start a new PWM session");
}

void testPwmTriggerExposesEmptySessionBeforeStart() {
    stereo3d::PWMTrigger trigger("gpiochip-test", 0, 90.0, 50.0);
    stereo3d::PWMTriggerEdgeEvent output;
    check(
        !trigger.eventForSequence(1, output),
        "unstarted PWM session must not fabricate trigger events");
    check(
        trigger.latestEventSequence() == 0,
        "unstarted PWM session must report sequence zero");
}

}  // namespace

int main() {
    testLookupIsStableAndBounded();
    testNonMonotonicEventsAreRejected();
    testPwmTriggerExposesEmptySessionBeforeStart();
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "trigger edge history tests passed\n";
    return EXIT_SUCCESS;
}
