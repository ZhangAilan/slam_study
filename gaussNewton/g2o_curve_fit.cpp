#include <Eigen/Core>
#include <Eigen/Dense>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

struct DataPoint {
    double x;
    double y;
};

class VertexParam : public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void setToOriginImpl() override
    {
        _estimate.setZero();
    }

    void oplusImpl(const double* update) override
    {
        _estimate += Eigen::Vector3d(update[0], update[1], update[2]);
    }

    bool read(std::istream& is) override
    {
        for (int i = 0; i < 3; ++i) {
            is >> _estimate[i];
        }
        return true;
    }

    bool write(std::ostream& os) const override
    {
        for (int i = 0; i < 3; ++i) {
            os << _estimate[i] << ' ';
        }
        return true;
    }
};

class ExponentialEdge : public g2o::BaseUnaryEdge<1, double, VertexParam> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ExponentialEdge(double x, double y)
        : x_(x), y_(y)
    {
        setMeasurement(y_);
        Eigen::Matrix<double, 1, 1> info;
        info.setIdentity();
        setInformation(info);
    }

    void computeError() override
    {
        const auto& p = static_cast<const VertexParam*>(_vertices[0])->estimate();
        const double exponent = p[0] * x_ * x_ + p[1] * x_ + p[2];
        const double y_hat = std::exp(exponent);
        _error[0] = y_ - y_hat;
    }

    void linearizeOplus() override
    {
        const auto& p = static_cast<const VertexParam*>(_vertices[0])->estimate();
        const double exponent = p[0] * x_ * x_ + p[1] * x_ + p[2];
        const double y_hat = std::exp(exponent);

        _jacobianOplusXi(0, 0) = -x_ * x_ * y_hat;
        _jacobianOplusXi(0, 1) = -x_ * y_hat;
        _jacobianOplusXi(0, 2) = -y_hat;
    }

    bool read(std::istream& is) override
    {
        is >> x_ >> y_;
        return true;
    }

    bool write(std::ostream& os) const override
    {
        os << x_ << ' ' << y_;
        return true;
    }

private:
    double x_;
    double y_;
};

int main()
{
    const Eigen::Vector3d trueParameters(-0.12, 0.25, 0.40);
    constexpr double noiseStdDev = 0.05;
    std::mt19937 rng(42);
    std::normal_distribution<double> gaussianNoise(0.0, noiseStdDev);

    std::vector<DataPoint> data;
    data.reserve(21);
    for (int i = 0; i <= 20; ++i) {
        const double x = -2.0 + 0.2 * i;
        const double exponent = trueParameters[0] * x * x
                              + trueParameters[1] * x
                              + trueParameters[2];
        const double y = std::exp(exponent) + gaussianNoise(rng);
        data.push_back({x, y});
    }

    g2o::SparseOptimizer optimizer;
    using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<3, 3>>;
    using LinearSolverType = g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;

    auto* solver = new g2o::OptimizationAlgorithmLevenberg(
        std::unique_ptr<BlockSolverType>(
            new BlockSolverType(std::unique_ptr<LinearSolverType>(new LinearSolverType))));
    optimizer.setAlgorithm(solver);

    auto* vertex = new VertexParam();
    vertex->setId(0);
    vertex->setEstimate(Eigen::Vector3d(-0.08, 0.20, 0.30));
    optimizer.addVertex(vertex);

    for (size_t i = 0; i < data.size(); ++i) {
        auto* edge = new ExponentialEdge(data[i].x, data[i].y);
        edge->setId(static_cast<int>(i));
        edge->setVertex(0, vertex);
        optimizer.addEdge(edge);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(100);

    const Eigen::Vector3d estimated = vertex->estimate();

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "真值参数: a = " << trueParameters[0]
              << ", b = " << trueParameters[1]
              << ", c = " << trueParameters[2] << '\n';
    std::cout << "高斯噪声标准差 = " << noiseStdDev
              << ", 观测点数 = " << data.size() << '\n';
    std::cout << "最终拟合参数：\n";
    std::cout << "a = " << estimated[0] << '\n';
    std::cout << "b = " << estimated[1] << '\n';
    std::cout << "c = " << estimated[2] << '\n';

    std::cout << "\n拟合参数与真值对比：\n";
    std::cout << "参数       真值           拟合值         绝对误差\n";
    for (int i = 0; i < 3; ++i) {
        std::cout << "abc"[i] << "          "
                  << trueParameters[i] << "   "
                  << estimated[i] << "   "
                  << std::abs(estimated[i] - trueParameters[i]) << '\n';
    }

    std::cout << "\n拟合模型：\ny = exp("
              << estimated[0] << " * x^2 + "
              << estimated[1] << " * x + "
              << estimated[2] << ")\n";

    return 0;
}
