#include "detect/trt_engine_contract.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

stereo3d::DetectorTensorSpec tensor(
    std::string name,
    bool input,
    std::vector<std::int64_t> dimensions,
    bool float32 = true) {
    return {std::move(name), input, float32, std::move(dimensions)};
}

std::vector<stereo3d::DetectorTensorSpec> validSingleOutput() {
    return {
        tensor("images", true, {1, 3, 640, 640}),
        tensor("output0", false, {1, 5, 8400}),
    };
}

std::vector<stereo3d::DetectorTensorSpec> validDflOutputs() {
    return {
        tensor("images", true, {1, 3, 640, 640}),
        tensor("cls8", false, {1, 80, 80, 1}),
        tensor("box8", false, {1, 80, 80, 64}),
        tensor("cls16", false, {1, 40, 40, 1}),
        tensor("box16", false, {1, 40, 40, 64}),
        tensor("cls32", false, {1, 20, 20, 1}),
        tensor("box32", false, {1, 20, 20, 64}),
    };
}

void testAcceptsSupportedContracts() {
    stereo3d::DetectorEngineContract contract;
    std::string error;
    check(stereo3d::validateDetectorEngineContract(
              validSingleOutput(), contract, error),
          "static single-output engine should be accepted");
    check(contract.input_name == "images" && contract.input_size == 640,
          "single-output contract must preserve input identity and size");
    check(contract.single_output_name == "output0" &&
              contract.single_output_elements == 42000 &&
              !contract.multi_scale_output,
          "single-output contract must derive a checked element count");

    check(stereo3d::validateDetectorEngineContract(
              validDflOutputs(), contract, error),
          "three-scale DFL engine should be accepted");
    check(contract.output_count == 6 && contract.multi_scale_output,
          "DFL contract must select multi-scale mode");
}

void testRejectsUnsupportedInput() {
    stereo3d::DetectorEngineContract contract;
    std::string error;

    auto tensors = validSingleOutput();
    tensors.front().dimensions = {1, 3, -1, -1};
    check(!stereo3d::validateDetectorEngineContract(tensors, contract, error),
          "dynamic input dimensions must be rejected");

    tensors = validSingleOutput();
    tensors.front().dimensions = {1, 3, 384, 640};
    check(!stereo3d::validateDetectorEngineContract(tensors, contract, error),
          "non-square input must be rejected");

    tensors = validSingleOutput();
    tensors.front().is_float32 = false;
    check(!stereo3d::validateDetectorEngineContract(tensors, contract, error),
          "non-float input must be rejected");

    tensors = validSingleOutput();
    tensors.insert(tensors.begin() + 1,
                   tensor("aux", true, {1, 3, 640, 640}));
    check(!stereo3d::validateDetectorEngineContract(tensors, contract, error),
          "multiple inputs must be rejected");
}

void testRejectsUnsupportedOutputs() {
    stereo3d::DetectorEngineContract contract;
    std::string error;

    auto tensors = validSingleOutput();
    tensors.push_back(tensor("unexpected", false, {1, 1}));
    check(!stereo3d::validateDetectorEngineContract(tensors, contract, error),
          "unbound extra output must be rejected");

    tensors = validDflOutputs();
    tensors.back().dimensions = {1, 20, 20, 32};
    check(!stereo3d::validateDetectorEngineContract(tensors, contract, error),
          "unsupported DFL channel count must be rejected");

    tensors = validDflOutputs();
    tensors.back().dimensions = {1, 40, 40, 64};
    check(!stereo3d::validateDetectorEngineContract(tensors, contract, error),
          "missing scale pairing must be rejected");

    tensors = validSingleOutput();
    tensors.back().dimensions = {1, -1, 6};
    check(!stereo3d::validateDetectorEngineContract(tensors, contract, error),
          "dynamic output dimensions must be rejected");

    tensors = validSingleOutput();
    tensors.back().dimensions = {1, 4, 8400};
    check(!stereo3d::validateDetectorEngineContract(tensors, contract, error),
          "pre-NMS output without a class score must be rejected");

    tensors = validSingleOutput();
    tensors.back().dimensions = {1, 42000};
    check(!stereo3d::validateDetectorEngineContract(tensors, contract, error),
          "single-output layout not understood by postprocess must be rejected");
}

}  // namespace

int main() {
    testAcceptsSupportedContracts();
    testRejectsUnsupportedInput();
    testRejectsUnsupportedOutputs();
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "TensorRT engine contract tests passed\n";
    return EXIT_SUCCESS;
}
