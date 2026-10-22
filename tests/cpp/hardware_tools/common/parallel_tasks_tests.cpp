#include "common/parallel_tasks.h"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAILED: " << message << '\n';
    }
}

void testEveryIndexRunsOnce() {
    std::vector<std::atomic<int>> visits(257);
    for (auto& visit : visits) visit.store(0);
    std::string error;
    check(volleyball::tools::hardware::runIndexedTasks(
              visits.size(),
              7,
              [&](std::size_t index) { visits[index].fetch_add(1); },
              error),
          "healthy indexed tasks must succeed");
    for (const auto& visit : visits) {
        check(visit.load() == 1, "every task index must run exactly once");
    }
    check(error.empty(), "successful run must not report an error");
}

void testTaskExceptionIsContained() {
    std::string error;
    check(!volleyball::tools::hardware::runIndexedTasks(
              32,
              4,
              [](std::size_t index) {
                  if (index == 3) throw std::runtime_error("bad image");
              },
              error),
          "task exception must become a failure result");
    check(error.find("bad image") != std::string::npos,
          "task exception text must remain diagnosable");
}

void testInvalidInvocationFailsClosed() {
    std::string error;
    check(!volleyball::tools::hardware::runIndexedTasks(
              1, 0, [](std::size_t) {}, error),
          "zero workers must be rejected");
    check(!volleyball::tools::hardware::runIndexedTasks(
              1, 1, volleyball::tools::hardware::IndexedTask{}, error),
          "empty task callback must be rejected");
    check(volleyball::tools::hardware::runIndexedTasks(
              0, 1, [](std::size_t) {}, error),
          "empty work set must finish successfully");
}

}  // namespace

int main() {
    testEveryIndexRunsOnce();
    testTaskExceptionIsContained();
    testInvalidInvocationFailsClosed();
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "parallel_tasks tests passed\n";
    return EXIT_SUCCESS;
}
