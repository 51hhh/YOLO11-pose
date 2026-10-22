#include "capture/async_grab_state.h"

#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>

namespace {

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void testRequestCompletesAcrossThreads() {
    stereo3d::AsyncGrabState state;
    state.start();

    auto worker = std::async(std::launch::async, [&state] {
        int slot = -1;
        if (!state.waitForRequest(slot) || slot != 1) return false;
        return state.complete(true);
    });

    check(state.request(1, 3), "valid request must be accepted");
    check(worker.wait_for(std::chrono::seconds(1)) ==
              std::future_status::ready,
          "grab worker must receive the request");
    check(worker.get(), "grab worker must publish completion");
    check(state.waitForCompletion(), "successful completion must be visible");

    state.stop();
}

void testStopDrainsAcceptedRequest() {
    stereo3d::AsyncGrabState state;
    state.start();
    check(state.request(2, 3), "request before stop must be accepted");

    state.stop();
    int slot = -1;
    check(state.waitForRequest(slot),
          "stop must preserve an already accepted request");
    check(slot == 2, "preserved request must keep its slot identity");
    check(state.complete(true), "preserved request must be completable");
    check(state.waitForCompletion(),
          "pipeline must observe completion after stop");

    slot = -1;
    check(!state.waitForRequest(slot),
          "worker must exit after draining stopped state");
}

void testStopRejectsNewAndOverlappingRequests() {
    stereo3d::AsyncGrabState state;
    state.start();
    check(state.request(0, 3), "first single-flight request must succeed");
    check(!state.request(1, 3), "overlapping request must be rejected");

    int slot = -1;
    check(state.waitForRequest(slot) && slot == 0,
          "worker must receive only the first request");
    check(state.complete(false), "failed grab must still complete handshake");
    check(!state.waitForCompletion(),
          "failed grab result must reach the pipeline");

    state.stop();
    check(!state.request(1, 3), "request after stop must be rejected");
    check(!state.request(-1, 3), "negative slot must be rejected");
    check(!state.request(3, 3), "out-of-range slot must be rejected");
}

}  // namespace

int main() {
    testRequestCompletesAcrossThreads();
    testStopDrainsAcceptedRequest();
    testStopRejectsNewAndOverlappingRequests();
    std::cout << "async grab state tests passed\n";
    return 0;
}
