#include "hybrid_astar_cpp/mpc_controller.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>

namespace {

inline double wrapAngle(double a) {
    return std::atan2(std::sin(a), std::cos(a));
}

inline double smoothstep(double edge0, double edge1, double x) {
    if (edge1 <= edge0) return 1.0;
    const double t = std::clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

}  // namespace

// ---------------------------------------------------------------------------
MpcController::MpcController(const Params& p)
    : params_(p),
      warm_start_(Eigen::VectorXd::Zero(p.horizon_N * 2)) {}

void MpcController::reset() {
    warm_start_.setZero();
}

// ---------------------------------------------------------------------------
// Dynamic bicycle model and finite-difference Jacobians.
// ---------------------------------------------------------------------------
MpcController::State MpcController::statePlusDelta(
    const State& x,
    const Eigen::Matrix<double, kStateDim, 1>& dx)
{
    State y = x;
    y.x += dx(0);
    y.y += dx(1);
    y.yaw = wrapAngle(y.yaw + dx(2));
    y.vx += dx(3);
    y.vy += dx(4);
    y.r += dx(5);
    return y;
}

Eigen::Matrix<double, MpcController::kStateDim, 1>
MpcController::dynamicsVector(const State& x, const Control& u) const {
    const double m = std::max(1.0, params_.mass);
    const double Iz = std::max(1.0, params_.yaw_inertia);
    const double lf = std::max(1e-3, params_.lf);
    const double lr = std::max(1e-3, params_.lr);
    const double vx_eps = std::max(1e-3, params_.vx_eps);
    const double slip_limit = std::abs(params_.slip_angle_limit);
    const double force_limit = std::abs(params_.tire_force_limit);
    const double a = std::clamp(u.a, params_.a_min, params_.a_max);
    const double delta = std::clamp(u.delta, params_.delta_min, params_.delta_max);

    const double denom = std::max(std::abs(x.vx), vx_eps);
    const double alpha_f =
        std::clamp(delta - std::atan2(x.vy + lf * x.r, denom),
                   -slip_limit, slip_limit);
    const double alpha_r =
        std::clamp(-std::atan2(x.vy - lr * x.r, denom),
                   -slip_limit, slip_limit);

    const double Fyf = std::clamp(params_.Cf * alpha_f, -force_limit, force_limit);
    const double Fyr = std::clamp(params_.Cr * alpha_r, -force_limit, force_limit);

    const double c = std::cos(x.yaw);
    const double s = std::sin(x.yaw);

    Eigen::Matrix<double, kStateDim, 1> dyn;
    dyn << x.vx * c - x.vy * s,
           x.vx * s + x.vy * c,
           x.r,
           a + x.vy * x.r,
           (Fyf + Fyr) / m - x.vx * x.r,
           (lf * Fyf - lr * Fyr) / Iz;

    const double low_speed_tau = 0.35;
    const double L = std::max(1e-3, lf + lr);
    const double target_r = x.vx * std::tan(delta) / L;

    Eigen::Matrix<double, kStateDim, 1> low;
    low << x.vx * c,
           x.vx * s,
           x.r,
           a,
           -x.vy / low_speed_tau,
           (target_r - x.r) / low_speed_tau;

    const double w = smoothstep(0.0, vx_eps, std::abs(x.vx));
    return (1.0 - w) * low + w * dyn;
}

Eigen::Matrix<double, MpcController::kStateDim, 1>
MpcController::stateError(const State& x, const State& ref) const {
    Eigen::Matrix<double, kStateDim, 1> e;
    e << x.x - ref.x,
         x.y - ref.y,
         wrapAngle(x.yaw - ref.yaw),
         x.vx - ref.vx,
         x.vy - ref.vy,
         x.r - ref.r;
    return e;
}

Eigen::Matrix<double, MpcController::kStateDim, 1>
MpcController::affineDefect(const State& ref_k, const Control& u_k,
                            const State& ref_next) const {
    const auto f = dynamicsVector(ref_k, u_k);
    const double dt = params_.dt;

    Eigen::Matrix<double, kStateDim, 1> c;
    c << ref_k.x + dt * f(0) - ref_next.x,
         ref_k.y + dt * f(1) - ref_next.y,
         wrapAngle(ref_k.yaw + dt * f(2) - ref_next.yaw),
         ref_k.vx + dt * f(3) - ref_next.vx,
         ref_k.vy + dt * f(4) - ref_next.vy,
         ref_k.r + dt * f(5) - ref_next.r;
    return c;
}

void MpcController::linearise(
    const State& x_r,
    const Control& u_r,
    Eigen::Matrix<double, kStateDim, kStateDim>& A_d,
    Eigen::Matrix<double, kStateDim, kControlDim>& B_d) const
{
    Eigen::Matrix<double, kStateDim, kStateDim> A_c =
        Eigen::Matrix<double, kStateDim, kStateDim>::Zero();
    Eigen::Matrix<double, kStateDim, kControlDim> B_c =
        Eigen::Matrix<double, kStateDim, kControlDim>::Zero();

    const std::array<double, kStateDim> x_eps = {
        1e-4, 1e-4, 1e-5, 1e-4, 1e-4, 1e-5
    };
    for (int i = 0; i < kStateDim; ++i) {
        Eigen::Matrix<double, kStateDim, 1> dx =
            Eigen::Matrix<double, kStateDim, 1>::Zero();
        dx(i) = x_eps[i];
        const auto fp = dynamicsVector(statePlusDelta(x_r, dx), u_r);
        const auto fm = dynamicsVector(statePlusDelta(x_r, -dx), u_r);
        A_c.col(i) = (fp - fm) / (2.0 * x_eps[i]);
    }

    const std::array<double, kControlDim> u_eps = {1e-4, 1e-5};
    for (int i = 0; i < kControlDim; ++i) {
        Control up = u_r;
        Control um = u_r;
        if (i == 0) {
            up.a += u_eps[i];
            um.a -= u_eps[i];
        } else {
            up.delta += u_eps[i];
            um.delta -= u_eps[i];
        }
        const auto fp = dynamicsVector(x_r, up);
        const auto fm = dynamicsVector(x_r, um);
        B_c.col(i) = (fp - fm) / (2.0 * u_eps[i]);
    }

    const double dt = params_.dt;
    A_d = Eigen::Matrix<double, kStateDim, kStateDim>::Identity() + dt * A_c;
    B_d = dt * B_c;
}

// ---------------------------------------------------------------------------
// Power iteration: estimate λ_max(H) for FISTA step size.
// 20 iterations is enough for a well-conditioned QP of this size.
// ---------------------------------------------------------------------------
double MpcController::spectralRadius(const Eigen::MatrixXd& H) const {
    Eigen::VectorXd v = Eigen::VectorXd::Random(H.cols());
    v.normalize();
    double lambda = 0.0;
    for (int i = 0; i < 30; ++i) {
        Eigen::VectorXd Hv = H * v;
        const double nrm = Hv.norm();
        if (nrm < 1e-12) return 1.0;
        v = Hv / nrm;
        lambda = v.dot(H * v);
    }
    return std::max(lambda, 1e-6);
}

// ---------------------------------------------------------------------------
// FISTA — Nesterov-accelerated projected gradient for a box-constrained QP.
//
//     min   ½·uᵀ H u + gᵀ u         s.t.   lb ≤ u ≤ ub        (H ≻ 0)
//
//  Pseudocode:
//     y_0 = u_0                 (warm start)
//     t_0 = 1
//     for k = 1, 2, ...
//         u_k  = proj_{[lb,ub]}( y_{k-1} − α · (H y_{k-1} + g) )
//         t_k  = (1 + √(1 + 4 t_{k-1}²)) / 2
//         y_k  = u_k + ((t_{k-1} − 1) / t_k) · (u_k − u_{k-1})
//
//  Convergence rate O(1/k²) — Beck & Teboulle (SIAM J. Imag. Sci. 2009).
// ---------------------------------------------------------------------------
int MpcController::solveBoxQP(const Eigen::MatrixXd& H,
                               const Eigen::VectorXd& g,
                               const Eigen::VectorXd& lb,
                               const Eigen::VectorXd& ub,
                               Eigen::VectorXd&       u_io) const {
    const double L  = spectralRadius(H);
    const double alpha = 1.0 / L;

    Eigen::VectorXd u      = u_io.cwiseMax(lb).cwiseMin(ub);   // project warm-start
    Eigen::VectorXd u_prev = u;
    Eigen::VectorXd y      = u;
    double t = 1.0;

    int iter = 0;
    for (iter = 0; iter < params_.max_iter; ++iter) {
        const Eigen::VectorXd grad = H * y + g;
        Eigen::VectorXd u_next = (y - alpha * grad).cwiseMax(lb).cwiseMin(ub);

        const double t_next = 0.5 * (1.0 + std::sqrt(1.0 + 4.0 * t * t));
        y = u_next + ((t - 1.0) / t_next) * (u_next - u);

        const double step_norm = (u_next - u).norm();
        u_prev = u;
        u      = u_next;
        t      = t_next;

        if (step_norm < params_.tol) { ++iter; break; }
    }

    u_io = u;
    return iter;
}

// ---------------------------------------------------------------------------
// Main solve — builds the condensed QP and calls FISTA.
// ---------------------------------------------------------------------------
MpcController::Result MpcController::solve(
    const State& x0,
    const std::vector<State>&   reference_x,
    const std::vector<Control>& reference_u)
{
    const auto t_start = std::chrono::steady_clock::now();

    Result R;
    const int N = params_.horizon_N;
    const int nx = kStateDim, nu = kControlDim;

    if (static_cast<int>(reference_x.size()) < N + 1 ||
        static_cast<int>(reference_u.size()) < N) {
        R.u.a = 0.0;  R.u.delta = 0.0;
        R.success = false;
        return R;
    }

    // 1. Initial error δx₀ = x₀ − x_{r,0}  (wrap yaw)
    const Eigen::Matrix<double, kStateDim, 1> dx0 =
        stateError(x0, reference_x[0]);

    // 2. Linearise at every reference knot
    std::vector<Eigen::Matrix<double, kStateDim, kStateDim>> A_d(N);
    std::vector<Eigen::Matrix<double, kStateDim, kControlDim>> B_d(N);
    std::vector<Eigen::Matrix<double, kStateDim, 1>> c_d(N);
    for (int k = 0; k < N; ++k) {
        linearise(reference_x[k], reference_u[k], A_d[k], B_d[k]);
        c_d[k] = affineDefect(reference_x[k], reference_u[k], reference_x[k + 1]);
    }

    // 3. Build condensation matrices  δX = Φ δx₀ + affine + Γ δU
    //    δX ∈ ℝ^{6N} stacks δx₁..δx_N   (δx₀ is known)
    //    δU ∈ ℝ^{2N} stacks δu₀..δu_{N-1}
    Eigen::MatrixXd Phi   = Eigen::MatrixXd::Zero(nx * N, nx);
    Eigen::MatrixXd Gamma = Eigen::MatrixXd::Zero(nx * N, nu * N);
    Eigen::VectorXd affine = Eigen::VectorXd::Zero(nx * N);

    // Row-by-row incremental construction.
    //   δx_{k+1} = A_d[k] · δx_k + B_d[k] · δu_k + c_d[k]
    //   Previous row block is at index k-1 (for k ≥ 1), or Identity for k=0.
    for (int k = 0; k < N; ++k) {
        if (k == 0) {
            Phi.block(0, 0, nx, nx)            = A_d[0];
            Gamma.block(0, 0, nx, nu)          = B_d[0];
            affine.segment(0, nx)              = c_d[0];
        } else {
            // δx_{k+1} = A_d[k] · δx_k + B_d[k] · δu_k + c_d[k]
            Phi.block(nx * k, 0, nx, nx) =
                A_d[k] * Phi.block(nx * (k - 1), 0, nx, nx);

            // Column j < k : propagate previous-row block by A_d[k]
            Gamma.block(nx * k, 0, nx, nu * k) =
                A_d[k] * Gamma.block(nx * (k - 1), 0, nx, nu * k);

            // Column j == k : directly B_d[k]
            Gamma.block(nx * k, nu * k, nx, nu) = B_d[k];

            affine.segment(nx * k, nx) =
                A_d[k] * affine.segment(nx * (k - 1), nx) + c_d[k];
        }
    }

    // 4. Build reference-tracking error for stage costs. The reference does
    //    not have to be a perfect dynamic rollout; affine carries that defect.

    Eigen::VectorXd constant_dx = Phi * dx0 + affine;

    // 5. Cost weights: block-diagonal Q̄ (stage + terminal), R̄ (stage)
    //    Q̄ rows:  δx₁..δx_{N-1} use Q,  δx_N uses P.
    Eigen::VectorXd Qbar_diag(nx * N);
    for (int k = 0; k < N - 1; ++k) {
        Qbar_diag.segment<kStateDim>(nx * k) << params_.q_x, params_.q_y,
                                                params_.q_yaw, params_.q_vx,
                                                params_.q_vy, params_.q_r;
    }
    // Terminal block
    Qbar_diag.segment<kStateDim>(nx * (N - 1)) << params_.p_x, params_.p_y,
                                                  params_.p_yaw, params_.p_vx,
                                                  params_.p_vy, params_.p_r;

    Eigen::VectorXd Rbar_diag(nu * N);
    for (int k = 0; k < N; ++k) {
        Rbar_diag.segment<kControlDim>(nu * k) << params_.r_a, params_.r_delta;
    }

    // 6. Condensed Hessian H and gradient g   (QP form:  min ½ δUᵀ H δU + gᵀ δU)
    //
    //    J  = (constant_dx + Γ δU)ᵀ Q̄ (constant_dx + Γ δU) + δUᵀ R̄ δU
    //       = δUᵀ (Γᵀ Q̄ Γ + R̄) δU  + 2·(Γᵀ Q̄ constant_dx)ᵀ δU + const
    //    →  H = 2 (Γᵀ Q̄ Γ + R̄) ,   g = 2 Γᵀ Q̄ constant_dx
    const Eigen::MatrixXd Qbar = Qbar_diag.asDiagonal();
    const Eigen::MatrixXd Rbar = Rbar_diag.asDiagonal();

    Eigen::MatrixXd H = 2.0 * (Gamma.transpose() * Qbar * Gamma + Rbar);
    // Symmetrise (numerical cleanup)
    H = 0.5 * (H + H.transpose());

    Eigen::VectorXd g = 2.0 * Gamma.transpose() * Qbar * constant_dx;

    // 7. Absolute input bounds converted to δU bounds:
    //    lb_k = u_min − u_{r,k} ,  ub_k = u_max − u_{r,k}
    Eigen::VectorXd lb(nu * N), ub(nu * N);
    for (int k = 0; k < N; ++k) {
        lb(nu * k    ) = params_.a_min     - reference_u[k].a;
        ub(nu * k    ) = params_.a_max     - reference_u[k].a;
        lb(nu * k + 1) = params_.delta_min - reference_u[k].delta;
        ub(nu * k + 1) = params_.delta_max - reference_u[k].delta;
    }

    // 8. Warm-start δU by shifting the previous solution one step forward
    Eigen::VectorXd dU = Eigen::VectorXd::Zero(nu * N);
    if (warm_start_.size() == nu * N) {
        dU.head(nu * (N - 1)) = warm_start_.tail(nu * (N - 1));
        // last entry left at zero
    }

    // 9. Solve the box-QP
    R.iterations = solveBoxQP(H, g, lb, ub, dU);
    warm_start_  = dU;

    // 10. Reconstruct δX for caller visualisation
    Eigen::VectorXd dX = constant_dx + Gamma * dU;

    // 11. Populate result
    R.predicted_u.resize(N);
    R.predicted_x.resize(N + 1);
    R.predicted_x[0] = x0;

    for (int k = 0; k < N; ++k) {
        R.predicted_u[k].a     = reference_u[k].a     + dU(nu * k    );
        R.predicted_u[k].delta = reference_u[k].delta + dU(nu * k + 1);

        R.predicted_x[k + 1].x   = reference_x[k + 1].x   + dX(nx * k    );
        R.predicted_x[k + 1].y   = reference_x[k + 1].y   + dX(nx * k + 1);
        R.predicted_x[k + 1].yaw = wrapAngle(reference_x[k + 1].yaw + dX(nx * k + 2));
        R.predicted_x[k + 1].vx  = reference_x[k + 1].vx  + dX(nx * k + 3);
        R.predicted_x[k + 1].vy  = reference_x[k + 1].vy  + dX(nx * k + 4);
        R.predicted_x[k + 1].r   = reference_x[k + 1].r   + dX(nx * k + 5);
    }

    R.u = R.predicted_u[0];

    // Cost (for telemetry / debugging)
    R.cost = (dX.transpose() * Qbar * dX + dU.transpose() * Rbar * dU)(0);

    const auto t_end = std::chrono::steady_clock::now();
    R.solve_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    R.success = true;
    return R;
}
