// Test: Tensor operations
#include <iostream>
#include <cassert>
#include <cmath>
#include "neuralcore/tensor.h"

using namespace neuralcore;

#define ASSERT_NEAR(a, b, tol) \
    do { if (std::abs((a) - (b)) > (tol)) { \
        std::cerr << "FAIL at line " << __LINE__ << ": " \
                  << (a) << " != " << (b) << std::endl; return 1; \
    }} while(0)

#define ASSERT_TRUE(cond) \
    do { if (!(cond)) { \
        std::cerr << "FAIL at line " << __LINE__ << std::endl; return 1; \
    }} while(0)

int main() {
    std::cout << "=== Tensor Tests ===" << std::endl;

    // Test: construction and element access
    {
        Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
        ASSERT_NEAR(t.at(0, 0), 1.0f, 1e-6);
        ASSERT_NEAR(t.at(1, 2), 6.0f, 1e-6);
        ASSERT_TRUE(t.size() == 6);
        ASSERT_TRUE(t.ndim() == 2);
        std::cout << "  [PASS] Construction & element access" << std::endl;
    }

    // Test: factory functions
    {
        auto z = Tensor::zeros({3, 3});
        ASSERT_NEAR(z.at(1, 1), 0.0f, 1e-6);

        auto o = Tensor::ones({2, 2});
        ASSERT_NEAR(o.at(0, 1), 1.0f, 1e-6);

        auto e = Tensor::eye(3);
        ASSERT_NEAR(e.at(0, 0), 1.0f, 1e-6);
        ASSERT_NEAR(e.at(0, 1), 0.0f, 1e-6);

        auto a = Tensor::arange(0, 5, 1);
        ASSERT_TRUE(a.size() == 5);
        ASSERT_NEAR(a.at(3), 3.0f, 1e-6);

        std::cout << "  [PASS] Factory functions" << std::endl;
    }

    // Test: arithmetic
    {
        Tensor a({2, 2}, {1, 2, 3, 4});
        Tensor b({2, 2}, {5, 6, 7, 8});
        auto c = a + b;
        ASSERT_NEAR(c.at(0, 0), 6.0f, 1e-6);
        ASSERT_NEAR(c.at(1, 1), 12.0f, 1e-6);

        auto d = a * b;
        ASSERT_NEAR(d.at(0, 0), 5.0f, 1e-6);

        auto e = a - b;
        ASSERT_NEAR(e.at(0, 0), -4.0f, 1e-6);

        auto f = a + 10.0f;
        ASSERT_NEAR(f.at(0, 0), 11.0f, 1e-6);

        std::cout << "  [PASS] Arithmetic" << std::endl;
    }

    // Test: broadcasting
    {
        Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
        Tensor b({1, 3}, {10, 20, 30});
        auto c = a + b;
        ASSERT_NEAR(c.at(0, 0), 11.0f, 1e-6);
        ASSERT_NEAR(c.at(1, 2), 36.0f, 1e-6);
        std::cout << "  [PASS] Broadcasting" << std::endl;
    }

    // Test: matmul
    {
        Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
        Tensor b({3, 2}, {7, 8, 9, 10, 11, 12});
        auto c = a.matmul(b);
        // [1,2,3]*[7,9,11] = 7+18+33 = 58
        ASSERT_NEAR(c.at(0, 0), 58.0f, 1e-6);
        ASSERT_NEAR(c.at(0, 1), 64.0f, 1e-6);
        std::cout << "  [PASS] Matmul" << std::endl;
    }

    // Test: reductions
    {
        Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
        ASSERT_NEAR(a.sum_all(), 21.0f, 1e-6);
        ASSERT_NEAR(a.mean_all(), 3.5f, 1e-6);
        ASSERT_NEAR(a.max_all(), 6.0f, 1e-6);
        ASSERT_NEAR(a.min_all(), 1.0f, 1e-6);

        auto s = a.sum(1, false);
        ASSERT_NEAR(s.at(0), 6.0f, 1e-6);
        ASSERT_NEAR(s.at(1), 15.0f, 1e-6);

        std::cout << "  [PASS] Reductions" << std::endl;
    }

    // Test: reshape & transpose
    {
        Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
        auto b = a.reshape({3, 2});
        ASSERT_NEAR(b.at(0, 0), 1.0f, 1e-6);
        ASSERT_NEAR(b.at(2, 1), 6.0f, 1e-6);

        auto c = a.t();
        ASSERT_TRUE(c.shape()[0] == 3 && c.shape()[1] == 2);
        ASSERT_NEAR(c.at(0, 1), 4.0f, 1e-6);

        std::cout << "  [PASS] Reshape & transpose" << std::endl;
    }

    // Test: math functions
    {
        Tensor a({2}, {1.0f, 4.0f});
        auto b = a.sqrt();
        ASSERT_NEAR(b.at(0), 1.0f, 1e-6);
        ASSERT_NEAR(b.at(1), 2.0f, 1e-6);

        auto c = a.exp();
        ASSERT_NEAR(c.at(0), std::exp(1.0f), 1e-5);

        auto d = a.log();
        ASSERT_NEAR(d.at(0), 0.0f, 1e-6);
        ASSERT_NEAR(d.at(1), std::log(4.0f), 1e-5);

        std::cout << "  [PASS] Math functions" << std::endl;
    }

    // Test: clone & contiguous
    {
        Tensor a({2, 2}, {1, 2, 3, 4});
        auto b = a.clone();
        b.at(0, 0) = 99.0f;
        ASSERT_NEAR(a.at(0, 0), 1.0f, 1e-6);
        ASSERT_NEAR(b.at(0, 0), 99.0f, 1e-6);
        std::cout << "  [PASS] Clone" << std::endl;
    }

    // Test: print
    {
        Tensor a({2, 2}, {1, 2, 3, 4});
        std::cout << "  Print test: " << a << std::endl;
        std::cout << "  [PASS] Print" << std::endl;
    }

    std::cout << "\nAll tensor tests passed!" << std::endl;
    return 0;
}
