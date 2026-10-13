
#include <Eigen/Dense>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// 观测数据
struct DataPoint {
    double x;
    double y;
};

// 计算残差向量 r 和雅可比矩阵 J
// 模型：y_hat = exp(a*x^2 + b*x + c)
// 残差：r = y - y_hat
bool computeResidualAndJacobian(
    const std::vector<DataPoint>& data,
    const VectorXd& theta,
    VectorXd& residual,
    MatrixXd& J)
{
    const int m = static_cast<int>(data.size());

    residual.resize(m);
    J.resize(m, 3);

    const double a = theta(0);
    const double b = theta(1);
    const double c = theta(2);

    for (int i = 0; i < m; ++i) {
        const double x = data[i].x;
        const double y = data[i].y;

        const double exponent = a * x * x + b * x + c;

        // 避免 exp 溢出或产生非有限值
        if (!std::isfinite(exponent) || exponent > 700.0) {
            return false;
        }

        const double y_hat = std::exp(exponent);

        if (!std::isfinite(y_hat)) {
            return false;
        }

        // 残差：r_i = y_i - y_hat_i
        residual(i) = y - y_hat;

        // 雅可比矩阵第 i 行：
        // [-x_i^2 * y_hat_i, -x_i * y_hat_i, -y_hat_i]
        J(i, 0) = -x * x * y_hat;
        J(i, 1) = -x * y_hat;
        J(i, 2) = -y_hat;
    }

    return residual.allFinite() && J.allFinite();
}

// 目标函数：F = 1/2 * r^T * r
double costFunction(const VectorXd& residual)
{
    return 0.5 * residual.squaredNorm();
}

int main()
{
    // 真值模型：y = exp(a*x^2 + b*x + c) + noise
    VectorXd trueTheta(3);
    trueTheta << -0.12, 0.25, 0.40;

    // 使用固定随机种子，便于重复实验；噪声服从 N(0, noiseStdDev^2)
    constexpr double noiseStdDev = 0.05;
    std::mt19937 rng(42);
    std::normal_distribution<double> gaussianNoise(0.0, noiseStdDev);

    std::vector<DataPoint> data;
    for (int i = 0; i <= 20; ++i) {
        const double x = -2.0 + 0.2 * i;
        const double exponent = trueTheta(0) * x * x
                              + trueTheta(1) * x
                              + trueTheta(2);
        const double y = std::exp(exponent) + gaussianNoise(rng);
        data.push_back({x, y});
    }

    if (data.size() < 3) {
        std::cerr << "错误：至少需要 3 个观测点。\n";
        return 1;
    }

    // 参数向量 theta = [a, b, c]^T
    // 初始参数猜测值
    VectorXd theta(3);
    theta << -0.08, 0.20, 0.30;

    const int maxIterations = 100;
    const double tolerance = 1e-8;
    const double minStepScale = 1e-8;

    VectorXd residual;
    MatrixXd J;

    if (!computeResidualAndJacobian(data, theta, residual, J)) {
        std::cerr << "初始参数无效，无法计算残差或雅可比矩阵。\n";
        return 1;
    }

    double cost = costFunction(residual);

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "真值参数: a = " << trueTheta(0)
              << ", b = " << trueTheta(1)
              << ", c = " << trueTheta(2) << '\n';
    std::cout << "高斯噪声标准差 = " << noiseStdDev
              << ", 观测点数 = " << data.size() << '\n';
    std::cout << "初始参数: a = " << theta(0)
              << ", b = " << theta(1)
              << ", c = " << theta(2) << '\n';
    std::cout << "初始 cost = " << cost << "\n\n";

    for (int iter = 0; iter < maxIterations; ++iter) {
        // 使用正规方程求解：J^T J * delta = -J^T * residual
        const MatrixXd H = J.transpose() * J;
        const VectorXd g = -J.transpose() * residual;
        Eigen::LDLT<MatrixXd> ldlt(H);

        if (ldlt.info() != Eigen::Success || !ldlt.isPositive()) {
            std::cerr << "正规方程矩阵不可分解或不是正定矩阵，参数无法可靠估计。\n";
            return 1;
        }

        const VectorXd delta = ldlt.solve(g);

        if (!delta.allFinite()) {
            std::cerr << "参数增量出现非有限值，迭代终止。\n";
            return 1;
        }

        // 回溯线搜索：如果完整更新不能降低 cost，
        // 就缩小步长，提升迭代的稳健性。
        double stepScale = 1.0;
        bool accepted = false;

        VectorXd thetaNew(3);
        VectorXd residualNew;
        MatrixXd JNew;
        double costNew = cost;

        while (stepScale >= minStepScale) {
            thetaNew = theta + stepScale * delta;

            if (computeResidualAndJacobian(
                    data, thetaNew, residualNew, JNew)) {

                costNew = costFunction(residualNew);

                if (std::isfinite(costNew) && costNew < cost) {
                    accepted = true;
                    break;
                }
            }

            stepScale *= 0.5;
        }

        if (!accepted) {
            // 若增量已经很小，可以视为收敛；
            // 否则当前方法无法找到下降步长。
            if (delta.norm() < tolerance * (1.0 + theta.norm())) {
                std::cout << "参数增量足够小，停止迭代。\n";
                break;
            }

            std::cout << "未找到能够降低目标函数的更新步长，停止迭代。\n";
            break;
        }

        const double actualStepNorm =
            (thetaNew - theta).norm();

        theta = thetaNew;
        residual = residualNew;
        J = JNew;

        const double previousCost = cost;
        cost = costNew;

        std::cout << "迭代 " << iter + 1
                  << ": cost = " << cost
                  << ", a = " << theta(0)
                  << ", b = " << theta(1)
                  << ", c = " << theta(2)
                  << ", step = " << stepScale << '\n';

        // 参数增量收敛
        if (actualStepNorm < tolerance * (1.0 + theta.norm())) {
            std::cout << "参数增量收敛，停止迭代。\n";
            break;
        }

        // 目标函数改善量很小
        if (previousCost - cost <
            tolerance * (1.0 + previousCost)) {
            std::cout << "目标函数改善量很小，停止迭代。\n";
            break;
        }
    }

    std::cout << "\n最终拟合参数：\n";
    std::cout << "a = " << theta(0) << '\n';
    std::cout << "b = " << theta(1) << '\n';
    std::cout << "c = " << theta(2) << '\n';
    std::cout << "最终 cost = " << cost << '\n';

    std::cout << "\n拟合参数与真值对比：\n";
    std::cout << "参数       真值           拟合值         绝对误差\n";
    for (int i = 0; i < 3; ++i) {
        std::cout << "abc"[i] << "          "
                  << trueTheta(i) << "   "
                  << theta(i) << "   "
                  << std::abs(theta(i) - trueTheta(i)) << '\n';
    }

    std::cout << "\n拟合模型：\n";
    std::cout << "y = exp("
              << theta(0) << " * x^2 + "
              << theta(1) << " * x + "
              << theta(2) << ")\n";

    return 0;
}
