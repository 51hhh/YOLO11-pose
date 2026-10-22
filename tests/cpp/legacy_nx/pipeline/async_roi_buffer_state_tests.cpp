#include "pipeline/async_roi_buffer_state.h"

#include <cstdlib>
#include <iostream>

namespace {

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void testNormalAcquireRelease() {
    stereo3d::AsyncRoiBufferState state;
    state.initializeFree();
    check(state.acquire(), "free buffer must be acquirable");
    check(!state.acquire(), "in-use buffer must not be acquired twice");
    check(state.release() ==
              stereo3d::AsyncRoiBufferTransition::ENQUEUE_FREE,
          "safe release must request one free-pool enqueue");
    check(state.release() == stereo3d::AsyncRoiBufferTransition::INVALID,
          "duplicate release must not enqueue twice");
}

void testDiagnosticReferenceDefersRelease() {
    stereo3d::AsyncRoiBufferState state;
    state.initializeFree();
    check(state.acquire(), "diagnostic test buffer must be acquired");
    check(state.addDiagnosticRef(), "first diagnostic ref must be accepted");
    check(state.addDiagnosticRef(), "second diagnostic ref must be accepted");
    check(state.release() == stereo3d::AsyncRoiBufferTransition::NONE,
          "main release must wait for diagnostic readers");
    check(state.ownership() ==
              stereo3d::AsyncRoiBufferOwnership::WAITING_DIAGNOSTIC,
          "buffer must expose diagnostic-wait ownership");
    check(state.releaseDiagnosticRef() ==
              stereo3d::AsyncRoiBufferTransition::NONE,
          "non-final diagnostic release must not enqueue");
    check(state.releaseDiagnosticRef() ==
              stereo3d::AsyncRoiBufferTransition::ENQUEUE_FREE,
          "final diagnostic release must enqueue exactly once");
}

void testEarlyDiagnosticCompletionStillNeedsMainRelease() {
    stereo3d::AsyncRoiBufferState state;
    state.initializeFree();
    check(state.acquire(), "buffer must be acquired");
    check(state.addDiagnosticRef(), "diagnostic ref must be accepted");
    check(state.releaseDiagnosticRef() ==
              stereo3d::AsyncRoiBufferTransition::NONE,
          "diagnostic completion cannot release a live main task");
    check(state.release() ==
              stereo3d::AsyncRoiBufferTransition::ENQUEUE_FREE,
          "main release must enqueue after early diagnostic completion");
}

void testQuarantineIsTerminalUntilReinitialize() {
    stereo3d::AsyncRoiBufferState state;
    state.initializeFree();
    check(state.acquire(), "quarantine test buffer must be acquired");
    check(state.addDiagnosticRef(), "diagnostic ref must be accepted");
    state.quarantine();
    check(state.isQuarantined(), "unsafe buffer must enter quarantine");
    check(state.release() == stereo3d::AsyncRoiBufferTransition::NONE,
          "quarantined main release must never enqueue");
    check(state.releaseDiagnosticRef() ==
              stereo3d::AsyncRoiBufferTransition::NONE,
          "quarantined diagnostic release must never enqueue");
    check(!state.acquire(), "quarantined buffer must not be reacquired");
    state.initializeFree();
    check(state.acquire(), "full reinitialization may restore the buffer");
}

}  // namespace

int main() {
    testNormalAcquireRelease();
    testDiagnosticReferenceDefersRelease();
    testEarlyDiagnosticCompletionStillNeedsMainRelease();
    testQuarantineIsTerminalUntilReinitialize();
    std::cout << "async ROI buffer state tests passed\n";
    return 0;
}
