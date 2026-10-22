#include "pipeline/pipeline_run_state.h"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <thread>

namespace {

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void testNormalStartStopCleanupOnce() {
    stereo3d::PipelineRunState state;
    check(state.isIdle(), "fresh state must be idle");
    check(state.begin(), "first begin must start a run");
    check(!state.isIdle(), "running state must not be idle");
    check(state.isRunning(), "begin must publish running state");
    check(!state.begin(), "second begin must not create another run");
    state.requestStop();
    check(!state.isRunning(), "stop request must stop new scheduling");
    check(state.claimCleanup(), "stopped run must retain cleanup responsibility");
    check(!state.claimCleanup(), "cleanup responsibility must be single-consumer");
    check(!state.begin(), "pipeline must not restart while cleanup is running");
    check(state.finishCleanup(), "cleanup owner must finish the transition");
    check(state.isIdle(), "normal cleanup must restore idle state");
}

void testFailureDoesNotLoseCleanupResponsibility() {
    stereo3d::PipelineRunState state;
    check(state.begin(), "failure test must start a run");
    state.markFailed();
    check(state.hasFailed(), "worker failure must be observable");
    check(!state.isRunning(), "worker failure must stop new scheduling");
    check(!state.begin(), "failed run must be cleaned before restart");
    check(state.claimCleanup(),
          "worker failure must preserve join/drain cleanup responsibility");
    check(state.finishCleanup(), "failed cleanup must reach a terminal phase");
}

void testNormalRestartButFailureIsTerminal() {
    stereo3d::PipelineRunState state;
    check(state.begin(), "first run must start");
    state.requestStop();
    check(state.claimCleanup(), "normal run must be cleaned");
    check(state.finishCleanup(), "normal cleanup must finish");
    check(state.begin(), "normally cleaned pipeline may start again");
    state.markFailed();
    check(state.claimCleanup(), "failed run must still be cleaned");
    check(state.finishCleanup(), "failed cleanup must finish");
    check(!state.begin(),
          "failed pipeline instance must be reconstructed before restart");
    check(state.hasFailed(), "failed state must remain observable");
}

void testConcurrentFailureAndStopHaveOneCleanupOwner() {
    for (int iteration = 0; iteration < 128; ++iteration) {
        stereo3d::PipelineRunState state;
        check(state.begin(), "concurrency test run must start");
        std::atomic<int> cleanup_owners{0};

        std::thread stopper([&] {
            state.requestStop();
            if (state.claimCleanup()) {
                cleanup_owners.fetch_add(1, std::memory_order_relaxed);
            }
        });
        std::thread failing_worker([&] {
            state.markFailed();
            state.requestStop();
            if (state.claimCleanup()) {
                cleanup_owners.fetch_add(1, std::memory_order_relaxed);
            }
        });
        stopper.join();
        failing_worker.join();

        check(cleanup_owners.load(std::memory_order_relaxed) == 1,
              "concurrent stop/failure must elect exactly one cleanup owner");
        check(state.finishCleanup(), "elected owner must finish cleanup");
        check(state.hasFailed(), "concurrent worker failure must remain visible");
    }
}

}  // namespace

int main() {
    testNormalStartStopCleanupOnce();
    testFailureDoesNotLoseCleanupResponsibility();
    testNormalRestartButFailureIsTerminal();
    testConcurrentFailureAndStopHaveOneCleanupOwner();
    std::cout << "pipeline run state tests passed\n";
    return 0;
}
