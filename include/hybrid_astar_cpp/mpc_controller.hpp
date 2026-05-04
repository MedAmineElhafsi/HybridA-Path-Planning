#ifndef HYBRID_ASTAR_CPP_MPC_CONTROLLER_HPP_
#define HYBRID_ASTAR_CPP_MPC_CONTROLLER_HPP_

#include <vector>
#include <Eigen/Dense>

// =============================================================================
// Linear Model Predictive Controller (MPC) — Ackermann kinematic vehicle
// -----------------------------------------------------------------------------
//
//   STATE        x = [X, Y, θ, v]ᵀ ∈ ℝ⁴           (rear-axle centre, world frame)
//   INPUT        u = [a, δ]ᵀ       ∈ ℝ²            (acceleration, steering angle)
//
//   CONTINUOUS DYNAMICS  ẋ = f(x,u)
//       ẋ = v·cos(θ)          ẏ = v·sin(θ)
//       θ̇ = (v/L)·tan(δ)      v̇ = a                      (L = wheelbase)
//
//   LINEARISATION at reference (x_r, u_r)
//       A_c = ∂f/∂x |_ref     B_c = ∂f/∂u |_ref            (see .cpp for entries)
//
//   DISCRETISATION  (forward-Euler, step dt)
//       A_d = I + dt·A_c         B_d = dt·B_c
//
//   ERROR-STATE DYNAMICS       δx = x−x_r,  δu = u−u_r
//       δx_{k+1} = A_{d,k}·δx_k + B_{d,k}·δu_k       (no affine term)
//
//   COST
//       J = Σ_{k=0}^{N-1} δx_kᵀ Q δx_k + δu_kᵀ R δu_k    + δx_Nᵀ P δx_N
//
//   CONDENSED QP               δX = Φ·δx_0 + Γ·δU
//       min  ½·δUᵀ H δU + gᵀ δU     H = 2(Γᵀ Q̄ Γ + R̄)
//        δU                          g = 2·Γᵀ Q̄ Φ δx_0
//       s.t. u_min − u_{r,k} ≤ δu_k ≤ u_max − u_{r,k}     (k = 0..N−1)
//
//   SOLVER   FISTA (Nesterov-accelerated projected gradient).
//            Step size α = 1/λ_max(H), convergence O(1/k²).
//            Warm-started from previous solution.
//
// REFERENCES
//   [1] Rawlings, Mayne, Diehl: "Model Predictive Control: Theory, Computation,
//       and Design", 2nd ed., Nob Hill, 2020. (Ch. 1-2 — nonlinear model +
//       linearisation; Ch. 8 — stability of tracking MPC)
//   [2] Kouvaritakis, Cannon: "Model Predictive Control: Classical, Robust and
//       Stochastic", Springer, 2016. (Ch. 2-3 — condensed formulation)
//   [3] Beck, Teboulle: "A Fast Iterative Shrinkage-Thresholding Algorithm for
//       Linear Inverse Problems", SIAM J. Imag. Sci. 2(1):183-202, 2009.
//       (FISTA — the projected-gradient solver used here)
//   [4] Kong, Pfeiffer, Schildbach, Borrelli: "Kinematic and Dynamic Vehicle
//       Models for Autonomous Driving Control Design", IV 2015.
//   [5] Falcone, Borrelli, Asgari, Tseng, Hrovat: "Predictive Active Steering
//       Control for Autonomous Vehicle Systems", IEEE Trans. CST, 2007.
// =============================================================================

class MpcController {
public:
    // -------------------------------------------------------------------------
    // Configuration — all knobs exposed for ROS 2 parameters.
    // -------------------------------------------------------------------------
    struct Params {
        int    horizon_N  = 20;     // prediction horizon (steps)
        double dt         = 0.1;    // MPC time step (s)
        double wheelbase  = 2.7;    // L (m)

        // Stage cost Q — diagonal weights [q_x, q_y, q_yaw, q_v]
        double q_x      = 5.0;
        double q_y      = 5.0;
        double q_yaw    = 3.0;
        double q_v      = 1.0;

        // Stage cost R — diagonal weights [r_a, r_delta]
        double r_a      = 0.1;
        double r_delta  = 1.0;

        // Terminal cost P — diagonal, typically heavier than Q
        double p_x      = 50.0;
        double p_y      = 50.0;
        double p_yaw    = 10.0;
        double p_v      = 5.0;

        // Input constraints (absolute — so MPC limits u, not δu)
        double a_min        = -3.0,  a_max     = 2.0;     // m/s²
        double delta_min    = -0.6,  delta_max = 0.6;     // rad (~34°)

        // FISTA solver
        int    max_iter = 200;
        double tol      = 1e-4;
    };

    struct State   { double x = 0, y = 0, yaw = 0, v = 0; };
    struct Control { double a = 0, delta = 0; };

    struct Result {
        Control                u;               // first optimal input — what to apply now
        std::vector<State>     predicted_x;     // predicted state sequence  (length N+1)
        std::vector<Control>   predicted_u;     // predicted input sequence (length N)
        int                    iterations = 0;
        double                 solve_time_ms = 0.0;
        double                 cost = 0.0;
        bool                   success = false;
    };

    explicit MpcController(const Params& p);

    // Solve one MPC step.
    //   x0            — current vehicle state (measured)
    //   reference_x   — desired state sequence, size must be ≥ N+1
    //   reference_u   — desired input sequence, size must be ≥ N
    Result solve(const State& x0,
                 const std::vector<State>&   reference_x,
                 const std::vector<Control>& reference_u);

    void reset();
    const Params& params() const { return params_; }

private:
    // Jacobian discretisation at a single reference step.
    void linearise(const State& x_r, const Control& u_r,
                   Eigen::Matrix4d&               A_d,
                   Eigen::Matrix<double, 4, 2>&   B_d) const;

    // Box-constrained QP:  min ½ uᵀH u + gᵀu   s.t.  lb ≤ u ≤ ub
    // Returns number of FISTA iterations used.
    int solveBoxQP(const Eigen::MatrixXd& H,
                   const Eigen::VectorXd& g,
                   const Eigen::VectorXd& lb,
                   const Eigen::VectorXd& ub,
                   Eigen::VectorXd&       u_io) const;

    // Estimate λ_max(H) via power iteration.
    double spectralRadius(const Eigen::MatrixXd& H) const;

    Params            params_;
    Eigen::VectorXd   warm_start_;   // previous δU for warm-starting FISTA
};

#endif  // HYBRID_ASTAR_CPP_MPC_CONTROLLER_HPP_
