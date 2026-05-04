#include "hybrid_astar_cpp/mpc_controller.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace {

inline double wrapAngle(double a) {
    return std::atan2(std::sin(a), std::cos(a));
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
// Continuous-time Jacobians, Euler discretisation.
//
//     A_c = [[0 0 -v_r sinθ_r   cosθ_r    ]
//            [0 0  v_r cosθ_r   sinθ_r    ]
//            [0 0    0           tanδ_r / L]
//            [0 0    0              0     ]]
//
//     B_c = [[0                     0                  ]
//            [0                     0                  ]
//            [0                     v_r / (L·cos²δ_r)  ]
//            [1                     0                  ]]
//
//     A_d = I + dt·A_c     B_d = dt·B_c
// ---------------------------------------------------------------------------
void MpcController::linearise(const State& x_r, const Control& u_r,
                               Eigen::Matrix4d&             A_d,
                               Eigen::Matrix<double, 4, 2>& B_d) const {
    const double L  = std::max(1e-3, params_.wheelbase);
    const double v  = x_r.v;
    const double th = x_r.yaw;
    const double d  = u_r.delta;

    const double cos_th = std::cos(th);
    const double sin_th = std::sin(th);
    const double cos_d  = std::cos(d);
    const double cos_d2 = std::max(1e-4, cos_d * cos_d);   // guard against δ≈±π/2

    Eigen::Matrix4d A_c = Eigen::Matrix4d::Zero();
    A_c(0, 2) = -v * sin_th;
    A_c(0, 3) =  cos_th;
    A_c(1, 2) =  v * cos_th;
    A_c(1, 3) =  sin_th;
    A_c(2, 3) =  std::tan(d) / L;

    Eigen::Matrix<double, 4, 2> B_c = Eigen::Matrix<double, 4, 2>::Zero();
    B_c(2, 1) = v / (L * cos_d2);     // ∂θ̇/∂δ
    B_c(3, 0) = 1.0;                  // ∂v̇/∂a

    const double dt = params_.dt;
    A_d = Eigen::Matrix4d::Identity() + dt * A_c;
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
    const int nx = 4, nu = 2;

    if (static_cast<int>(reference_x.size()) < N + 1 ||
        static_cast<int>(reference_u.size()) < N) {
        R.u.a = 0.0;  R.u.delta = 0.0;
        R.success = false;
        return R;
    }

    // 1. Initial error δx₀ = x₀ − x_{r,0}  (wrap yaw)
    Eigen::Vector4d dx0;
    dx0(0) = x0.x   - reference_x[0].x;
    dx0(1) = x0.y   - reference_x[0].y;
    dx0(2) = wrapAngle(x0.yaw - reference_x[0].yaw);
    dx0(3) = x0.v   - reference_x[0].v;

    // 2. Linearise at every reference knot
    std::vector<Eigen::Matrix4d>              A_d(N);
    std::vector<Eigen::Matrix<double, 4, 2>>  B_d(N);
    for (int k = 0; k < N; ++k) {
        linearise(reference_x[k], reference_u[k], A_d[k], B_d[k]);
    }

    // 3. Build condensation matrices  δX = Φ δx₀ + Γ δU
    //    δX ∈ ℝ^{4N} stacks δx₁..δx_N   (δx₀ is known)
    //    δU ∈ ℝ^{2N} stacks δu₀..δu_{N-1}
    Eigen::MatrixXd Phi   = Eigen::MatrixXd::Zero(nx * N, nx);
    Eigen::MatrixXd Gamma = Eigen::MatrixXd::Zero(nx * N, nu * N);

    // Row-by-row incremental construction.
    //   δx_{k+1} = A_d[k] · δx_k + B_d[k] · δu_k
    //   Previous row block is at index k-1 (for k ≥ 1), or Identity for k=0.
    for (int k = 0; k < N; ++k) {
        if (k == 0) {
            Phi.block(0, 0, nx, nx)            = A_d[0];
            Gamma.block(0, 0, nx, nu)          = B_d[0];
        } else {
            // δx_{k+1} = A_d[k] · δx_k + B_d[k] · δu_k
            Phi.block(nx * k, 0, nx, nx) =
                A_d[k] * Phi.block(nx * (k - 1), 0, nx, nx);

            // Column j < k : propagate previous-row block by A_d[k]
            Gamma.block(nx * k, 0, nx, nu * k) =
                A_d[k] * Gamma.block(nx * (k - 1), 0, nx, nu * k);

            // Column j == k : directly B_d[k]
            Gamma.block(nx * k, nu * k, nx, nu) = B_d[k];
        }
    }

    // 4. Build reference-tracking error for stage costs
    //    target_δX[k] = x_{r,k+1} − x_{r,k+1} = 0  (we track reference exactly)
    //    But we must carry x₀ − x_{r,0} through via Φ δx₀; already done.
    //    So we want to minimise ‖δX‖_Q̄² + ‖δU‖_R̄².

    Eigen::VectorXd Phi_dx0 = Phi * dx0;     // constant contribution in δX

    // 5. Cost weights: block-diagonal Q̄ (stage + terminal), R̄ (stage)
    //    Q̄ rows:  δx₁..δx_{N-1} use Q,  δx_N uses P.
    Eigen::VectorXd Qbar_diag(nx * N);
    for (int k = 0; k < N - 1; ++k) {
        Qbar_diag.segment<4>(nx * k) << params_.q_x, params_.q_y,
                                        params_.q_yaw, params_.q_v;
    }
    // Terminal block
    Qbar_diag.segment<4>(nx * (N - 1)) << params_.p_x, params_.p_y,
                                          params_.p_yaw, params_.p_v;

    Eigen::VectorXd Rbar_diag(nu * N);
    for (int k = 0; k < N; ++k) {
        Rbar_diag.segment<2>(nu * k) << params_.r_a, params_.r_delta;
    }

    // 6. Condensed Hessian H and gradient g   (QP form:  min ½ δUᵀ H δU + gᵀ δU)
    //
    //    J  = (Φ δx₀ + Γ δU)ᵀ Q̄ (Φ δx₀ + Γ δU) + δUᵀ R̄ δU
    //       = δUᵀ (Γᵀ Q̄ Γ + R̄) δU  + 2·(Γᵀ Q̄ Φ δx₀)ᵀ δU + const
    //    →  H = 2 (Γᵀ Q̄ Γ + R̄) ,   g = 2 Γᵀ Q̄ Φ δx₀
    const Eigen::MatrixXd Qbar = Qbar_diag.asDiagonal();
    const Eigen::MatrixXd Rbar = Rbar_diag.asDiagonal();

    Eigen::MatrixXd H = 2.0 * (Gamma.transpose() * Qbar * Gamma + Rbar);
    // Symmetrise (numerical cleanup)
    H = 0.5 * (H + H.transpose());

    Eigen::VectorXd g = 2.0 * Gamma.transpose() * Qbar * Phi_dx0;

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
    Eigen::VectorXd dX = Phi_dx0 + Gamma * dU;

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
        R.predicted_x[k + 1].v   = reference_x[k + 1].v   + dX(nx * k + 3);
    }

    R.u = R.predicted_u[0];

    // Cost (for telemetry / debugging)
    R.cost = (dX.transpose() * Qbar * dX + dU.transpose() * Rbar * dU)(0);

    const auto t_end = std::chrono::steady_clock::now();
    R.solve_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    R.success = true;
    return R;
}
