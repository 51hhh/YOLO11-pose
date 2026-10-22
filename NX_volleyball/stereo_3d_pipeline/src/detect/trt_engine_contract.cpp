#include "trt_engine_contract.h"

#include <climits>
#include <map>
#include <set>
#include <sstream>
#include <utility>

namespace stereo3d {
namespace {

bool positiveDimensions(const DetectorTensorSpec& tensor, std::string& error) {
    if (tensor.dimensions.empty()) {
        error = "tensor '" + tensor.name + "' has no dimensions";
        return false;
    }
    for (const std::int64_t dimension : tensor.dimensions) {
        if (dimension <= 0) {
            error = "tensor '" + tensor.name +
                    "' has dynamic or non-positive dimensions";
            return false;
        }
    }
    return true;
}

bool checkedElementCount(const DetectorTensorSpec& tensor,
                         int& elements,
                         std::string& error) {
    std::int64_t product = 1;
    for (const std::int64_t dimension : tensor.dimensions) {
        if (product > INT_MAX / dimension) {
            error = "tensor '" + tensor.name +
                    "' is too large for the current buffer index type";
            return false;
        }
        product *= dimension;
    }
    elements = static_cast<int>(product);
    return true;
}

}  // namespace

bool validateDetectorEngineContract(
    const std::vector<DetectorTensorSpec>& tensors,
    DetectorEngineContract& contract,
    std::string& error) {
    contract = DetectorEngineContract{};
    error.clear();

    if (tensors.empty()) {
        error = "engine exposes no I/O tensors";
        return false;
    }

    std::set<std::string> names;
    const DetectorTensorSpec* input = nullptr;
    std::vector<const DetectorTensorSpec*> outputs;
    for (const auto& tensor : tensors) {
        if (tensor.name.empty()) {
            error = "engine exposes an unnamed I/O tensor";
            return false;
        }
        if (!names.insert(tensor.name).second) {
            error = "engine exposes duplicate tensor name '" + tensor.name + "'";
            return false;
        }
        if (!tensor.is_float32) {
            error = "tensor '" + tensor.name + "' is not float32";
            return false;
        }
        if (!positiveDimensions(tensor, error)) {
            return false;
        }
        if (tensor.is_input) {
            if (input != nullptr) {
                error = "engine must expose exactly one input tensor";
                return false;
            }
            input = &tensor;
        } else {
            outputs.push_back(&tensor);
        }
    }

    if (input == nullptr) {
        error = "engine must expose exactly one input tensor";
        return false;
    }
    if (input->dimensions.size() != 4 || input->dimensions[0] != 1 ||
        input->dimensions[1] != 3 || input->dimensions[2] != input->dimensions[3] ||
        input->dimensions[2] > INT_MAX) {
        error = "input tensor '" + input->name +
                "' must be static float32 NCHW [1,3,S,S]";
        return false;
    }

    contract.input_name = input->name;
    contract.input_size = static_cast<int>(input->dimensions[2]);
    contract.output_count = static_cast<int>(outputs.size());

    if (outputs.size() == 1) {
        const DetectorTensorSpec& output = *outputs.front();
        const auto& dims = output.dimensions;
        if (dims.size() != 3 || dims[0] != 1) {
            error = "single YOLO output '" + output.name +
                    "' must have shape [1,C,N] or [1,N,C]";
            return false;
        }
        const bool post_nms = dims[2] == 6 && dims[1] <= 1000;
        const std::int64_t channels = dims[1] < dims[2] ? dims[1] : dims[2];
        if (!post_nms && channels < 5) {
            error = "pre-NMS YOLO output '" + output.name +
                    "' must contain four box values and at least one class score";
            return false;
        }
        contract.single_output_name = output.name;
        if (!checkedElementCount(output,
                                 contract.single_output_elements, error)) {
            return false;
        }
        return true;
    }

    if (outputs.size() != 6) {
        std::ostringstream message;
        message << "engine exposes " << outputs.size()
                << " outputs; current detector supports exactly 1 or 6";
        error = message.str();
        return false;
    }

    struct ScaleMembers {
        int classification = 0;
        int regression = 0;
    };
    std::map<std::pair<std::int64_t, std::int64_t>, ScaleMembers> scales;
    for (const DetectorTensorSpec* output : outputs) {
        const auto& dims = output->dimensions;
        if (dims.size() != 4 || dims[0] != 1 || dims[1] != dims[2] ||
            contract.input_size % dims[1] != 0) {
            error = "DFL tensor '" + output->name +
                    "' must be static NHWC [1,H,H,C] and divide input size";
            return false;
        }
        auto& members = scales[{dims[1], dims[2]}];
        if (dims[3] == 1) {
            ++members.classification;
        } else if (dims[3] == 64) {
            ++members.regression;
        } else {
            error = "DFL tensor '" + output->name +
                    "' must have 1 classification or 64 regression channels";
            return false;
        }
    }
    if (scales.size() != 3) {
        error = "six-output DFL engine must expose exactly three spatial scales";
        return false;
    }
    for (const auto& item : scales) {
        if (item.second.classification != 1 || item.second.regression != 1) {
            error = "each DFL scale must expose one classification and one regression tensor";
            return false;
        }
    }

    contract.multi_scale_output = true;
    return true;
}

}  // namespace stereo3d
