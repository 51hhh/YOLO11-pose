#include "volleyball/adapters/gpio/pwm_trigger.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAILED: " << message << '\n';
    }
}

static_assert(std::is_base_of_v<
              volleyball::ports::TriggerController,
              volleyball::adapters::gpio::PWMTrigger>);

volleyball::adapters::gpio::PWMTriggerEdgeEvent event(std::uint64_t sequence) {
    return {
        sequence,
        1000 + sequence * 100,
        1010 + sequence * 100,
        2000 + sequence * 100,
        20,
    };
}

void testLookupIsStableAndBounded() {
    volleyball::adapters::gpio::PWMTriggerEventHistory history(3);
    history.push(event(1));
    history.push(event(2));
    history.push(event(3));
    volleyball::adapters::gpio::PWMTriggerEdgeEvent output;
    check(history.find(2, output), "retained trigger must be found");
    check(output.edge_realtime_ns == 2200, "trigger fields must remain unchanged");
    history.push(event(4));
    check(!history.find(1, output), "oldest trigger must leave bounded history");
    check(history.find(4, output), "latest trigger must be found");
    check(history.latestSequence() == 4, "latest sequence must be auditable");
    check(history.size() == 3, "history must honor its capacity");
}

void testNonMonotonicEventsAreRejected() {
    volleyball::adapters::gpio::PWMTriggerEventHistory history;
    history.push(event(2));
    history.push(event(2));
    history.push(event(1));
    check(history.size() == 1, "duplicate and reversed events must be ignored");
    history.clear();
    check(history.size() == 0, "clear must start a new PWM session");
}

void testPwmTriggerExposesEmptySessionBeforeStart() {
    volleyball::adapters::gpio::PWMTrigger trigger(
        "gpiochip-test", 0, 90.0, 50.0);
    volleyball::adapters::gpio::PWMTriggerEdgeEvent output;
    check(
        !trigger.eventForSequence(1, output),
        "unstarted PWM session must not fabricate trigger events");
    check(
        trigger.latestEventSequence() == 0,
        "unstarted PWM session must report sequence zero");
}

void testBuildWithoutGpiodFailsClosed() {
#ifdef NO_GPIOD
    volleyball::adapters::gpio::PWMTrigger trigger(
        "gpiochip-test", 0, 90.0, 50.0);
    std::string error;
    check(!trigger.start(error),
          "a build without libgpiod must not use an implicit GPIO backend");
    check(!error.empty(),
          "missing libgpiod support must return an explicit startup error");
#endif
}

void testPwmTimingRejectsNonFiniteAndUnrepresentableValues() {
    using volleyball::adapters::gpio::PWMTrigger;
    check(PWMTrigger::isTimingValid(90.0, 50.0),
          "normal camera trigger timing must be valid");
    check(PWMTrigger::isTimingValid(1.0, 0.000001),
          "a duty cycle with non-zero nanosecond phases must be valid");

    const double infinity = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    check(!PWMTrigger::isTimingValid(0.0, 50.0),
          "zero frequency must be rejected");
    check(!PWMTrigger::isTimingValid(-1.0, 50.0),
          "negative frequency must be rejected");
    check(!PWMTrigger::isTimingValid(infinity, 50.0),
          "infinite frequency must be rejected");
    check(!PWMTrigger::isTimingValid(nan, 50.0),
          "NaN frequency must be rejected");
    check(!PWMTrigger::isTimingValid(90.0, infinity),
          "infinite duty cycle must be rejected");
    check(!PWMTrigger::isTimingValid(90.0, nan),
          "NaN duty cycle must be rejected");
    check(!PWMTrigger::isTimingValid(90.0, 0.0),
          "zero duty cycle has no observable rising pulse");
    check(!PWMTrigger::isTimingValid(90.0, 100.0),
          "full duty cycle has no repeated rising pulse");
    check(!PWMTrigger::isTimingValid(1e9, 50.0),
          "one-nanosecond period cannot represent both phases");
    check(!PWMTrigger::isTimingValid(1e-20, 50.0),
          "period outside long nanosecond range must be rejected");
}

}  // namespace

int main() {
    testLookupIsStableAndBounded();
    testNonMonotonicEventsAreRejected();
    testPwmTriggerExposesEmptySessionBeforeStart();
    testBuildWithoutGpiodFailsClosed();
    testPwmTimingRejectsNonFiniteAndUnrepresentableValues();
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "trigger edge history tests passed\n";
    return EXIT_SUCCESS;
}
