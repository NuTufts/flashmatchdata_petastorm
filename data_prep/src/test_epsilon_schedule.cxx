#include <iostream>
#include <vector>
#include <iomanip>
#include "UnbalancedSinkhornDivergence.h"

using namespace flashmatch::sinkhorn;

int main() {
    std::cout << "Testing epsilon_schedule implementation" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // Test parameters matching Python test
    double p = 2.0;
    double diameter = 403.1863;
    double blur = 1.0;
    double scaling = 0.5;

    std::cout << "Parameters:" << std::endl;
    std::cout << "  p = " << p << std::endl;
    std::cout << "  diameter = " << diameter << std::endl;
    std::cout << "  blur = " << blur << std::endl;
    std::cout << "  scaling = " << scaling << std::endl;
    std::cout << std::endl;

    auto eps_list = UnbalancedSinkhornDivergence::epsilon_schedule(p, diameter, blur, scaling);

    std::cout << "Generated " << eps_list.size() << " epsilon values:" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < eps_list.size(); ++i) {
        std::cout << "  " << std::setw(2) << i << ": "
                  << std::setw(12) << eps_list[i] << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Expected output from Python geomloss:" << std::endl;
    std::cout << "  0: 162559.192508" << std::endl;
    std::cout << "  1: 162559.192508" << std::endl;
    std::cout << "  2:  40639.798127" << std::endl;
    std::cout << "  3:  10159.949532" << std::endl;
    std::cout << "  4:   2539.987383" << std::endl;
    std::cout << "  5:    634.996846" << std::endl;
    std::cout << "  6:    158.749211" << std::endl;
    std::cout << "  7:     39.687303" << std::endl;
    std::cout << "  8:      9.921826" << std::endl;
    std::cout << "  9:      2.480456" << std::endl;
    std::cout << " 10:      1.000000" << std::endl;
    std::cout << std::endl;

    // Check if results match
    std::vector<double> expected = {
        162559.192508, 162559.192508, 40639.798127, 10159.949532,
        2539.987383, 634.996846, 158.749211, 39.687303,
        9.921826, 2.480456, 1.000000
    };

    bool all_match = true;
    double tolerance = 1e-3;  // Allow small numerical differences

    if (eps_list.size() != expected.size()) {
        std::cout << "ERROR: Size mismatch! Got " << eps_list.size()
                  << " values, expected " << expected.size() << std::endl;
        all_match = false;
    } else {
        for (size_t i = 0; i < eps_list.size(); ++i) {
            double diff = std::abs(eps_list[i] - expected[i]);
            double rel_error = diff / expected[i];
            if (rel_error > tolerance) {
                std::cout << "ERROR at index " << i << ": got " << eps_list[i]
                          << ", expected " << expected[i]
                          << " (rel error: " << rel_error << ")" << std::endl;
                all_match = false;
            }
        }
    }

    if (all_match) {
        std::cout << "SUCCESS: All values match Python geomloss implementation!" << std::endl;
    } else {
        std::cout << "FAILURE: Values don't match!" << std::endl;
        return 1;
    }

    return 0;
}