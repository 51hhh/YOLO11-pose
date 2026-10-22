#include "volleyball/logging/log.hpp"

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>

#include <unistd.h>

namespace {

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

std::string captureStderr(const std::function<void()>& emit) {
    int pipe_fds[2] = {-1, -1};
    check(pipe(pipe_fds) == 0, "创建 stderr 捕获管道失败");

    std::fflush(stderr);
    const int saved_stderr = dup(STDERR_FILENO);
    check(saved_stderr >= 0, "复制 stderr 失败");
    check(dup2(pipe_fds[1], STDERR_FILENO) >= 0, "重定向 stderr 失败");
    check(close(pipe_fds[1]) == 0, "关闭重复写端失败");

    emit();
    std::fflush(stderr);

    check(dup2(saved_stderr, STDERR_FILENO) >= 0, "恢复 stderr 失败");
    check(close(saved_stderr) == 0, "关闭 stderr 副本失败");

    std::string output;
    char chunk[4096];
    while (true) {
        const ssize_t count = read(pipe_fds[0], chunk, sizeof(chunk));
        if (count == 0) break;
        check(count > 0, "读取 stderr 捕获内容失败");
        output.append(chunk, static_cast<std::size_t>(count));
    }
    check(close(pipe_fds[0]) == 0, "关闭捕获读端失败");
    return output;
}

}  // namespace

int main() {
    const std::string normal = captureStderr([] {
        LOG_INFO("frame=%d status=%s", 42, "accepted");
    });
    check(normal.find("[INFO ] frame=42 status=accepted") != std::string::npos,
          "普通格式化日志内容不完整");
    check(!normal.empty() && normal.back() == '\n',
          "普通日志必须以换行结束");

    const std::string oversized(8192, 'x');
    const std::string truncated = captureStderr([&oversized] {
        LOG_ERROR("%s", oversized.c_str());
    });
    check(!truncated.empty(), "超长日志不应丢失整行");
    check(truncated.size() <= 2047, "超长日志越过固定缓冲区边界");
    check(truncated.back() == '\n', "截断日志必须仍以换行结束");
    check(truncated.find("[ERROR] ") != std::string::npos,
          "截断日志应保留级别前缀");

    std::cout << "logging_tests: ok\n";
    return 0;
}
