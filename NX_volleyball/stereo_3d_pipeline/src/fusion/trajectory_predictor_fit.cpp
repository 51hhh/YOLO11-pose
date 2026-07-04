#include "trajectory_predictor.h"

#include <cmath>

namespace stereo3d {

bool TrajectoryPredictor::fitQuadratic(
    const std::vector<double>& t, const std::vector<double>& y,
    double& a, double& b, double& c) {

    int n = (int)t.size();
    if (n < 3) return false;

    double S0 = n, S1 = 0, S2 = 0, S3 = 0, S4 = 0;
    double Sy = 0, Sty = 0, St2y = 0;
    for (int i = 0; i < n; i++) {
        double ti = t[i], ti2 = ti * ti, yi = y[i];
        S1 += ti; S2 += ti2; S3 += ti2 * ti; S4 += ti2 * ti2;
        Sy += yi; Sty += ti * yi; St2y += ti2 * yi;
    }

    // Solve 3x3 normal equation with Cramer's rule.
    double D = S4 * (S2 * S0 - S1 * S1) -
               S3 * (S3 * S0 - S1 * S2) +
               S2 * (S3 * S1 - S2 * S2);
    if (std::abs(D) < 1e-12) return false;

    a = (St2y * (S2 * S0 - S1 * S1) -
         S3 * (Sty * S0 - S1 * Sy) +
         S2 * (Sty * S1 - S2 * Sy)) / D;
    b = (S4 * (Sty * S0 - S1 * Sy) -
         St2y * (S3 * S0 - S1 * S2) +
         S2 * (S3 * Sy - Sty * S2)) / D;
    c = (S4 * (S2 * Sy - Sty * S1) -
         S3 * (S3 * Sy - Sty * S2) +
         St2y * (S3 * S1 - S2 * S2)) / D;
    return true;
}

bool TrajectoryPredictor::fitLinear(
    const std::vector<double>& t, const std::vector<double>& y,
    double& a, double& b) {

    int n = (int)t.size();
    if (n < 2) return false;

    double St = 0, Sy = 0, Stt = 0, Sty = 0;
    for (int i = 0; i < n; i++) {
        St += t[i]; Sy += y[i];
        Stt += t[i] * t[i]; Sty += t[i] * y[i];
    }
    double D = n * Stt - St * St;
    if (std::abs(D) < 1e-12) return false;

    a = (n * Sty - St * Sy) / D;
    b = (Stt * Sy - St * Sty) / D;
    return true;
}

}  // namespace stereo3d
