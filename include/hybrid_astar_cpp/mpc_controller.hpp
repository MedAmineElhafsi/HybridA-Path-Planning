#ifndef HYBRID_ASTAR_CPP_MPC_CONTROLLER_HPP_
#define HYBRID_ASTAR_CPP_MPC_CONTROLLER_HPP_

#include <vector>
#include <Eigen/Dense>

// =============================================================================
// Linear Model Predictive Controller (MPC) — dynamic bicycle vehicle
// -----------------------------------------------------------------------------
//
//   STATE        x = [X, Y, θ, vx, vy, r]ᵀ ∈ ℝ⁶
//                X,Y,θ are in the world frame; vx,vy,r are body-frame dynamics.
//   INPUT        u = [a, δ]ᵀ ∈ ℝ²                 (longitudinal acceleration,
//                                                  front steering angle)
//
//   CONTINUOUS DYNAMICS  ẋ = f(x,u), linear tire bicycle model
//       X_dot   = vx·cos(θ) - vy·sin(θ)
//       Y_dot   = vx·sin(θ) + vy·cos(θ)
//       θ_dot   = r
//       vx_dot  = a + vy·r
//       vy_dot  = (Fyf + Fyr)/m - vx·r
//       r_dot   = (lf·Fyf - lr·Fyr)/Iz
//
//       αf = δ - atan2(vy + lf·r, max(|vx|, vx_eps))
//       αr =    - atan2(vy - lr·r, max(|vx|, vx_eps))
//       Fyf = Cf·αf,  Fyr = Cr·αr
//
//   LINEARISATION at reference (x_r, u_r)
//       A_c = ∂f/∂x |_ref     B_c = ∂f/∂u |_ref
//       computed with centred finite differences so clamp/low-speed guards are
//       reflected in the local model.
//
//   DISCRETISATION  (forward-Euler, step dt)
//       A_d = I + dt·A_c         B_d = dt·B_c
//       c_d = x_r,k + dt·f(x_r,k,u_r,k) − x_r,k+1
//
//   ERROR-STATE DYNAMICS       δx = x−x_r,  δu = u−u_r
//       δx_{k+1} = A_{d,k}·δx_k + B_{d,k}·δu_k + c_{d,k}
//
//   COST
//       J = Σ_{k=0}^{N-1} δx_kᵀ Q δx_k + δu_kᵀ R δu_k    + δx_Nᵀ P δx_N
//
//   CONDENSED QP               δX = Φ·δx_0 + c̄ + Γ·δU
//       min  ½·δUᵀ H δU + gᵀ δU     H = 2(Γᵀ Q̄ Γ + R̄)
//        δU                          g = 2·Γᵀ Q̄ (Φ·δx_0 + c̄)
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
        double wheelbase  = 2.7;    // planner/reference wheelbase (m)

        // Dynamic bicycle plant parameters.
        double mass        = 1500.0;  // kg
        double yaw_inertia = 2250.0;  // kg*m^2
        double lf          = 1.2;     // front axle distance from CG (m)
        double lr          = 1.5;     // rear axle distance from CG (m)
        double Cf          = 60000.0; // front cornering stiffness (N/rad)
        double Cr          = 60000.0; // rear cornering stiffness (N/rad)
        double vx_eps      = 0.5;     // slip denominator guard (m/s)
        double slip_angle_limit = 0.5;    // rad
        double tire_force_limit = 8000.0; // N

        // Stage cost Q — diagonal weights [q_x, q_y, q_yaw, q_vx, q_vy, q_r]
        double q_x      = 5.0;
        double q_y      = 5.0;
        double q_yaw    = 3.0;
        double q_vx     = 1.0;
        double q_vy     = 0.5;
        double q_r      = 0.5;

        // Stage cost R — diagonal weights [r_a, r_delta]
        double r_a      = 0.1;
        double r_delta  = 1.0;

        // Terminal cost P — diagonal, typically heavier than Q
        double p_x      = 50.0;
        double p_y      = 50.0;
        double p_yaw    = 10.0;
        double p_vx     = 5.0;
        double p_vy     = 2.0;
        double p_r      = 2.0;

        // Input constraints (absolute — so MPC limits u, not δu)
        double a_min        = -3.0,  a_max     = 2.0;     // m/s²
        double delta_min    = -0.6,  delta_max = 0.6;     // rad (~34°)

        // FISTA solver
        int    max_iter = 200;
        double tol      = 1e-4;
    };

    struct State {
        double x = 0.0;
        double y = 0.0;
        double yaw = 0.0;
        double vx = 0.0;
        double vy = 0.0;
        double r = 0.0;
    };
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
    static constexpr int kStateDim = 6;
    static constexpr int kControlDim = 2;

    // Jacobian discretisation at a single reference step.
    void linearise(const State& x_r, const Control& u_r,
                   Eigen::Matrix<double, kStateDim, kStateDim>& A_d,
                   Eigen::Matrix<double, kStateDim, kControlDim>& B_d) const;

    Eigen::Matrix<double, kStateDim, 1>
    dynamicsVector(const State& x, const Control& u) const;

    Eigen::Matrix<double, kStateDim, 1>
    stateError(const State& x, const State& ref) const;

    Eigen::Matrix<double, kStateDim, 1>
    affineDefect(const State& ref_k, const Control& u_k,
                 const State& ref_next) const;

    static State statePlusDelta(
        const State& x,
        const Eigen::Matrix<double, kStateDim, 1>& dx);

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
