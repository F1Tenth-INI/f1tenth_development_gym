/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * Licensed under the 2-Clause BSD License.
 */

#ifndef ACADOS_SOLVER_ocp_safe_mpc_5293571f4164_01395577_H_
#define ACADOS_SOLVER_ocp_safe_mpc_5293571f4164_01395577_H_

#include "acados/utils/types.h"

#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

#define OCP_SAFE_MPC_5293571F4164_01395577_NX     8
#define OCP_SAFE_MPC_5293571F4164_01395577_NZ     0
#define OCP_SAFE_MPC_5293571F4164_01395577_NU     2
#define OCP_SAFE_MPC_5293571F4164_01395577_NP     0
#define OCP_SAFE_MPC_5293571F4164_01395577_NP_GLOBAL     0
#define OCP_SAFE_MPC_5293571F4164_01395577_NBX    2
#define OCP_SAFE_MPC_5293571F4164_01395577_NBX0   8
#define OCP_SAFE_MPC_5293571F4164_01395577_NBU    2
#define OCP_SAFE_MPC_5293571F4164_01395577_NSBX   0
#define OCP_SAFE_MPC_5293571F4164_01395577_NSBU   0
#define OCP_SAFE_MPC_5293571F4164_01395577_NSH    10
#define OCP_SAFE_MPC_5293571F4164_01395577_NSH0   0
#define OCP_SAFE_MPC_5293571F4164_01395577_NSG    0
#define OCP_SAFE_MPC_5293571F4164_01395577_NSPHI  0
#define OCP_SAFE_MPC_5293571F4164_01395577_NSHN   10
#define OCP_SAFE_MPC_5293571F4164_01395577_NSGN   0
#define OCP_SAFE_MPC_5293571F4164_01395577_NSPHIN 0
#define OCP_SAFE_MPC_5293571F4164_01395577_NSPHI0 0
#define OCP_SAFE_MPC_5293571F4164_01395577_NSBXN  0
#define OCP_SAFE_MPC_5293571F4164_01395577_NS     10
#define OCP_SAFE_MPC_5293571F4164_01395577_NS0    0
#define OCP_SAFE_MPC_5293571F4164_01395577_NSN    10
#define OCP_SAFE_MPC_5293571F4164_01395577_NG     1
#define OCP_SAFE_MPC_5293571F4164_01395577_NBXN   0
#define OCP_SAFE_MPC_5293571F4164_01395577_NGN    0
#define OCP_SAFE_MPC_5293571F4164_01395577_NY0    0
#define OCP_SAFE_MPC_5293571F4164_01395577_NY     0
#define OCP_SAFE_MPC_5293571F4164_01395577_NYN    2
#define OCP_SAFE_MPC_5293571F4164_01395577_N      30
#define OCP_SAFE_MPC_5293571F4164_01395577_NH     13
#define OCP_SAFE_MPC_5293571F4164_01395577_NHN    10
#define OCP_SAFE_MPC_5293571F4164_01395577_NH0    3
#define OCP_SAFE_MPC_5293571F4164_01395577_NPHI0  0
#define OCP_SAFE_MPC_5293571F4164_01395577_NPHI   0
#define OCP_SAFE_MPC_5293571F4164_01395577_NPHIN  0
#define OCP_SAFE_MPC_5293571F4164_01395577_NR     0

#ifdef __cplusplus
extern "C" {
#endif


// ** capsule for solver data **
typedef struct ocp_safe_mpc_5293571f4164_01395577_solver_capsule
{
    // acados objects
    ocp_nlp_in *nlp_in;
    ocp_nlp_out *nlp_out;
    ocp_nlp_out *sens_out;
    ocp_nlp_solver *nlp_solver;
    void *nlp_opts;
    ocp_nlp_plan_t *nlp_solver_plan;
    ocp_nlp_config *nlp_config;
    ocp_nlp_dims *nlp_dims;

    // number of expected runtime parameters
    unsigned int nlp_np;

    /* external functions */

    // dynamics

    external_function_external_param_casadi *discr_dyn_phi_fun;
    external_function_external_param_casadi *discr_dyn_phi_fun_jac_ut_xt;



    external_function_external_param_casadi *discr_dyn_phi_fun_jac_ut_xt_hess;


    // cost

    external_function_external_param_casadi *ext_cost_fun;
    external_function_external_param_casadi *ext_cost_fun_jac;
    external_function_external_param_casadi *ext_cost_fun_jac_hess;






    external_function_external_param_casadi ext_cost_0_fun;
    external_function_external_param_casadi ext_cost_0_fun_jac;
    external_function_external_param_casadi ext_cost_0_fun_jac_hess;





    external_function_external_param_casadi cost_y_e_fun;
    external_function_external_param_casadi cost_y_e_fun_jac_ut_xt;
    external_function_external_param_casadi cost_y_e_hess;


    // constraints
    external_function_external_param_casadi *nl_constr_h_fun_jac;
    external_function_external_param_casadi *nl_constr_h_fun;
    external_function_external_param_casadi *nl_constr_h_fun_jac_hess;






    external_function_external_param_casadi nl_constr_h_0_fun_jac;
    external_function_external_param_casadi nl_constr_h_0_fun;
    external_function_external_param_casadi nl_constr_h_0_fun_jac_hess;






    external_function_external_param_casadi nl_constr_h_e_fun_jac;
    external_function_external_param_casadi nl_constr_h_e_fun;
    external_function_external_param_casadi nl_constr_h_e_fun_jac_hess;




} ocp_safe_mpc_5293571f4164_01395577_solver_capsule;

ACADOS_SYMBOL_EXPORT ocp_safe_mpc_5293571f4164_01395577_solver_capsule * ocp_safe_mpc_5293571f4164_01395577_acados_create_capsule(void);
ACADOS_SYMBOL_EXPORT int ocp_safe_mpc_5293571f4164_01395577_acados_free_capsule(ocp_safe_mpc_5293571f4164_01395577_solver_capsule *capsule);

ACADOS_SYMBOL_EXPORT int ocp_safe_mpc_5293571f4164_01395577_acados_create(ocp_safe_mpc_5293571f4164_01395577_solver_capsule * capsule);

ACADOS_SYMBOL_EXPORT int ocp_safe_mpc_5293571f4164_01395577_acados_reset(ocp_safe_mpc_5293571f4164_01395577_solver_capsule* capsule, int reset_qp_solver_mem, int reset_numerical_values, int reset_solver_options, int reset_x_to_x0_bar);

/**
 * Generic version of ocp_safe_mpc_5293571f4164_01395577_acados_create which allows to use a different number of shooting intervals than
 * the number used for code generation. If new_time_steps=NULL and n_time_steps matches the number used for code
 * generation, the time-steps from code generation is used.
 */
ACADOS_SYMBOL_EXPORT int ocp_safe_mpc_5293571f4164_01395577_acados_create_with_discretization(ocp_safe_mpc_5293571f4164_01395577_solver_capsule * capsule, int n_time_steps, double* new_time_steps);
/**
 * Update the time step vector. Number N must be identical to the currently set number of shooting nodes in the
 * nlp_solver_plan. Returns 0 if no error occurred and a otherwise a value other than 0.
 */
ACADOS_SYMBOL_EXPORT int ocp_safe_mpc_5293571f4164_01395577_acados_update_time_steps(ocp_safe_mpc_5293571f4164_01395577_solver_capsule * capsule, int N, double* new_time_steps);
/**
 * This function is used for updating an already initialized solver with a different number of qp_cond_N.
 */
ACADOS_SYMBOL_EXPORT int ocp_safe_mpc_5293571f4164_01395577_acados_update_qp_solver_cond_N(ocp_safe_mpc_5293571f4164_01395577_solver_capsule * capsule, int qp_solver_cond_N);
ACADOS_SYMBOL_EXPORT int ocp_safe_mpc_5293571f4164_01395577_acados_update_params(ocp_safe_mpc_5293571f4164_01395577_solver_capsule * capsule, int stage, double *value, int np);
ACADOS_SYMBOL_EXPORT int ocp_safe_mpc_5293571f4164_01395577_acados_update_params_sparse(ocp_safe_mpc_5293571f4164_01395577_solver_capsule * capsule, int stage, int *idx, double *p, int n_update);
ACADOS_SYMBOL_EXPORT int ocp_safe_mpc_5293571f4164_01395577_acados_set_p_global_and_precompute_dependencies(ocp_safe_mpc_5293571f4164_01395577_solver_capsule* capsule, double* data, int data_len);

ACADOS_SYMBOL_EXPORT int ocp_safe_mpc_5293571f4164_01395577_acados_solve(ocp_safe_mpc_5293571f4164_01395577_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int ocp_safe_mpc_5293571f4164_01395577_acados_setup_qp_matrices_and_factorize(ocp_safe_mpc_5293571f4164_01395577_solver_capsule* capsule);



ACADOS_SYMBOL_EXPORT int ocp_safe_mpc_5293571f4164_01395577_acados_free(ocp_safe_mpc_5293571f4164_01395577_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void ocp_safe_mpc_5293571f4164_01395577_acados_print_stats(ocp_safe_mpc_5293571f4164_01395577_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int ocp_safe_mpc_5293571f4164_01395577_acados_custom_update(ocp_safe_mpc_5293571f4164_01395577_solver_capsule* capsule, double* data, int data_len);

ACADOS_SYMBOL_EXPORT ocp_nlp_in *ocp_safe_mpc_5293571f4164_01395577_acados_get_nlp_in(ocp_safe_mpc_5293571f4164_01395577_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *ocp_safe_mpc_5293571f4164_01395577_acados_get_nlp_out(ocp_safe_mpc_5293571f4164_01395577_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *ocp_safe_mpc_5293571f4164_01395577_acados_get_sens_out(ocp_safe_mpc_5293571f4164_01395577_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_solver *ocp_safe_mpc_5293571f4164_01395577_acados_get_nlp_solver(ocp_safe_mpc_5293571f4164_01395577_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_config *ocp_safe_mpc_5293571f4164_01395577_acados_get_nlp_config(ocp_safe_mpc_5293571f4164_01395577_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void *ocp_safe_mpc_5293571f4164_01395577_acados_get_nlp_opts(ocp_safe_mpc_5293571f4164_01395577_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_dims *ocp_safe_mpc_5293571f4164_01395577_acados_get_nlp_dims(ocp_safe_mpc_5293571f4164_01395577_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_plan_t *ocp_safe_mpc_5293571f4164_01395577_acados_get_nlp_plan(ocp_safe_mpc_5293571f4164_01395577_solver_capsule * capsule);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_SOLVER_ocp_safe_mpc_5293571f4164_01395577_H_
