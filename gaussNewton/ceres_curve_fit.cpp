#include <ceres/ceres.h>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

struct DataPoint {
    double x;
    double y;
};

// 残差模型：r = y_observed - exp(a*x^2 + b*x + c)
struct ExponentialResidual {
    ExponentialResidual(double x, double observedY)
        : x_(x), observedY_(observedY) {}

    template <typename T>
    bool operator()(const T* const parameters, T* residual) const
    {
        using std::exp;
        const T exponent = parameters[0] * T(x_ * x_)
                         + parameters[1] * T(x_)
                         + parameters[2];
        residual[0] = T(observedY_) - exp(exponent);
        return true;
    }

    const double x_;
    const double observedY_;
};

int main()
{
    // 真值模型：y = exp(a*x^2 + b*x + c) + Gaussian noise
    const std::array<double, 3> trueParameters = {-0.12, 0.25, 0.40};
    constexpr double noiseStdDev = 0.05;
    std::mt19937 rng(42);  // 固定种子，保证每次运行可复现
    std::normal_distribution<double> gaussianNoise(0.0, noiseStdDev);

    std::vector<DataPoint> data;
    data.reserve(21);
    for (int i = 0; i <= 20; ++i) {
        const double x = -2.0 + 0.2 * i;
        const double exponent = trueParameters[0] * x * x
                              + trueParameters[1] * x
                              + trueParameters[2];
        const double observedY = std::exp(exponent) + gaussianNoise(rng);
        data.push_back({x, observedY});
    }

    // 待优化参数 [a, b, c] 的初值
    double parameters[3] = {-0.08, 0.20, 0.30};

    ceres::Problem problem;
    for (const DataPoint& point : data) {
        auto* costFunction =
            new ceres::AutoDiffCostFunction<ExponentialResidual, 1, 3>(
                new ExponentialResidual(point.x, point.y));
        problem.AddResidualBlock(costFunction, nullptr, parameters);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "真值参数: a = " << trueParameters[0]
              << ", b = " << trueParameters[1]
              << ", c = " << trueParameters[2] << '\n';
    std::cout << "高斯噪声标准差 = " << noiseStdDev
              << ", 观测点数 = " << data.size() << '\n';
    std::cout << "初始 cost = " << summary.initial_cost << '\n';
    std::cout << "最终 cost = " << summary.final_cost << '\n';
    std::cout << "求解状态: " << summary.BriefReport() << '\n';

    std::cout << "\n拟合参数与真值对比：\n";
    std::cout << "参数       真值           拟合值         绝对误差\n";
    const char* parameterNames[] = {"a", "b", "c"};
    for (int i = 0; i < 3; ++i) {
        std::cout << parameterNames[i] << "          "
                  << trueParameters[i] << "   "
                  << parameters[i] << "   "
                  << std::abs(parameters[i] - trueParameters[i]) << '\n';
    }

    std::cout << "\n拟合模型：\ny = exp("
              << parameters[0] << " * x^2 + "
              << parameters[1] << " * x + "
              << parameters[2] << ")\n";

    return summary.IsSolutionUsable() ? 0 : 1;
}
