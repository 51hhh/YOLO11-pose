#include "utils/capture_v21_stage_schema.h"

#include <cstdlib>
#include <iostream>
#include <sstream>
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

std::string lineFrom(std::ostringstream& output) {
    std::string line = output.str();
    if (!line.empty() && line.back() == '\n') line.pop_back();
    return line;
}

std::vector<std::string> splitCsv(const std::string& line) {
    std::vector<std::string> fields;
    std::size_t start = 0;
    while (true) {
        const std::size_t comma = line.find(',', start);
        if (comma == std::string::npos) {
            fields.push_back(line.substr(start));
            return fields;
        }
        fields.push_back(line.substr(start, comma - start));
        start = comma + 1;
    }
}

}  // namespace

int main() {
    check(stereo3d::kCaptureV21StageSchemaVersion == 4,
          "stage schema version must remain explicit");

    {
        std::ostringstream header;
        stereo3d::writeCaptureV21FrameHeader(header);
        check(
            lineFrom(header) ==
                "frame_id,host_capture_timestamp_ns,callback_timestamp_ns,"
                "left_receive_monotonic_ns,right_receive_monotonic_ns,"
                "left_receive_realtime_ns,right_receive_realtime_ns,"
                "left_sdk_host_timestamp_raw,right_sdk_host_timestamp_raw,"
                "left_device_timestamp_ns,right_device_timestamp_ns,"
                "stereo_timestamp_residual_ns,trigger_sequence,"
                "trigger_scheduled_monotonic_ns,trigger_edge_monotonic_ns,"
                "trigger_edge_realtime_ns,trigger_write_uncertainty_ns,"
                "left_frame_number,right_frame_number,left_frame_counter,"
                "right_frame_counter,left_trigger_index,right_trigger_index,"
                "raw_left,raw_right",
            "frame header must match the Python staging contract");

        stereo3d::CaptureV21FrameRow row;
        row.frame_id = 1;
        row.host_capture_timestamp_ns = 2;
        row.callback_timestamp_ns = 3;
        row.left_receive_monotonic_ns = 4;
        row.right_receive_monotonic_ns = 5;
        row.left_receive_realtime_ns = 6;
        row.right_receive_realtime_ns = 7;
        row.left_sdk_host_timestamp_raw = 8;
        row.right_sdk_host_timestamp_raw = 9;
        row.left_device_timestamp_ns = 10;
        row.right_device_timestamp_ns = 11;
        row.stereo_timestamp_residual_ns = -12;
        row.trigger_sequence = 13;
        row.trigger_scheduled_monotonic_ns = 14;
        row.trigger_edge_monotonic_ns = 15;
        row.trigger_edge_realtime_ns = 16;
        row.trigger_write_uncertainty_ns = 17;
        row.left_frame_number = 18;
        row.right_frame_number = 19;
        row.left_frame_counter = 20;
        row.right_frame_counter = 21;
        row.left_trigger_index = 22;
        row.right_trigger_index = 23;
        row.raw_left = "raw/left/00000001.pgm";
        row.raw_right = "raw/right/00000001.pgm";
        std::ostringstream output;
        check(stereo3d::writeCaptureV21FrameRow(output, row),
              "valid frame row must serialize");
        const auto fields = splitCsv(lineFrom(output));
        const std::vector<std::string> expected = {
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11",
            "-12", "13", "14", "15", "16", "17", "18", "19", "20", "21",
            "22", "23", row.raw_left, row.raw_right,
        };
        check(fields == expected,
              "every frame value must remain aligned with its named column");

        row.raw_left = "raw,left.pgm";
        std::ostringstream invalid;
        check(!stereo3d::writeCaptureV21FrameRow(invalid, row),
              "unescaped CSV delimiter in a path must be rejected");
    }

    {
        std::ostringstream header;
        stereo3d::writeCaptureV21DetectionHeader(header);
        check(lineFrom(header) ==
                  "frame_id,camera_role,candidate_id,cx_px,cy_px,width_px,"
                  "height_px,confidence,class_id,selected",
              "detection header must match the Python staging contract");

        stereo3d::CaptureV21DetectionRow row;
        row.frame_id = 5;
        row.camera_role = "right";
        row.candidate_id = 7;
        row.cx_px = 10.25F;
        row.cy_px = 11.5F;
        row.width_px = 12.75F;
        row.height_px = 13.0F;
        row.confidence = 0.875F;
        row.class_id = 2;
        row.selected = true;
        std::ostringstream output;
        check(stereo3d::writeCaptureV21DetectionRow(output, row),
              "valid detection row must serialize");
        const auto fields = splitCsv(lineFrom(output));
        check(fields.size() == stereo3d::kCaptureV21DetectionColumns.size(),
              "detection row width must match its header");
        check(fields[1] == "right" && fields[9] == "true",
              "detection role and selection must keep their columns");
    }

    {
        std::ostringstream header;
        stereo3d::writeCaptureV21PairHeader(header);
        check(lineFrom(header) ==
                  "pair_id,frame_id,left_candidate_id,right_candidate_id,"
                  "disparity_px,epipolar_error_px,pair_score,"
                  "semantic_confidence,x_rig_m,y_rig_m,z_rig_m,accepted,"
                  "reject_reason",
              "pair header must match the Python staging contract");

        stereo3d::CaptureV21PairRow accepted;
        accepted.pair_id = 1;
        accepted.frame_id = 2;
        accepted.left_candidate_id = 3;
        accepted.right_candidate_id = 4;
        accepted.disparity_px = 5.5F;
        accepted.epipolar_error_px = 0.25F;
        accepted.pair_score = 0.125F;
        accepted.semantic_confidence = 0.9375F;
        accepted.position_rig_m = std::array<float, 3>{1.0F, 2.0F, 3.0F};
        accepted.accepted = true;
        std::ostringstream accepted_output;
        check(stereo3d::writeCaptureV21PairRow(accepted_output, accepted),
              "accepted pair with position must serialize");
        const auto accepted_fields = splitCsv(lineFrom(accepted_output));
        check(accepted_fields.size() == stereo3d::kCaptureV21PairColumns.size(),
              "accepted pair row width must match its header");
        check(accepted_fields[8] == "1" && accepted_fields[10] == "3" &&
                  accepted_fields[11] == "true" && accepted_fields[12].empty(),
              "accepted pair must carry position and no reject reason");

        stereo3d::CaptureV21PairRow rejected = accepted;
        rejected.position_rig_m.reset();
        rejected.accepted = false;
        rejected.reject_reason = "epipolar_reject";
        std::ostringstream rejected_output;
        check(stereo3d::writeCaptureV21PairRow(rejected_output, rejected),
              "rejected pair with reason must serialize");
        const auto rejected_fields = splitCsv(lineFrom(rejected_output));
        check(rejected_fields.size() == stereo3d::kCaptureV21PairColumns.size(),
              "rejected pair row width must match its header");
        check(rejected_fields[8].empty() && rejected_fields[9].empty() &&
                  rejected_fields[10].empty() && rejected_fields[11] == "false" &&
                  rejected_fields[12] == "epipolar_reject",
              "rejected pair must omit position and carry its reason");

        rejected.reject_reason.clear();
        std::ostringstream invalid;
        check(!stereo3d::writeCaptureV21PairRow(invalid, rejected),
              "rejected pair without a reason must be rejected");
    }

    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "capture v2.1 stage schema tests passed\n";
    return EXIT_SUCCESS;
}
