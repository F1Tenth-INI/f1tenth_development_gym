#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# Licensed under the 2-Clause BSD License.

cimport acados_solver_common

cdef extern from "acados_solver_ocp_safe_mpc_f5099c06c437_ee23731e.h":
    ctypedef struct nlp_solver_capsule "ocp_safe_mpc_f5099c06c437_ee23731e_solver_capsule":
        pass

    nlp_solver_capsule * acados_create_capsule "ocp_safe_mpc_f5099c06c437_ee23731e_acados_create_capsule"()
    int acados_free_capsule "ocp_safe_mpc_f5099c06c437_ee23731e_acados_free_capsule"(nlp_solver_capsule *capsule)

    int acados_create "ocp_safe_mpc_f5099c06c437_ee23731e_acados_create"(nlp_solver_capsule * capsule)

    int acados_create_with_discretization "ocp_safe_mpc_f5099c06c437_ee23731e_acados_create_with_discretization"(nlp_solver_capsule * capsule, int n_time_steps, double* new_time_steps)
    int acados_update_time_steps "ocp_safe_mpc_f5099c06c437_ee23731e_acados_update_time_steps"(nlp_solver_capsule * capsule, int N, double* new_time_steps)
    int acados_update_qp_solver_cond_N "ocp_safe_mpc_f5099c06c437_ee23731e_acados_update_qp_solver_cond_N"(nlp_solver_capsule * capsule, int qp_solver_cond_N)

    int acados_update_params "ocp_safe_mpc_f5099c06c437_ee23731e_acados_update_params"(nlp_solver_capsule * capsule, int stage, double *value, int np_)
    int acados_update_params_sparse "ocp_safe_mpc_f5099c06c437_ee23731e_acados_update_params_sparse"(nlp_solver_capsule * capsule, int stage, int *idx, double *p, int n_update)
    int acados_set_p_global_and_precompute_dependencies "ocp_safe_mpc_f5099c06c437_ee23731e_acados_set_p_global_and_precompute_dependencies"(nlp_solver_capsule * capsule, double *value, int data_len)
    int acados_solve "ocp_safe_mpc_f5099c06c437_ee23731e_acados_solve"(nlp_solver_capsule * capsule)
    int acados_reset "ocp_safe_mpc_f5099c06c437_ee23731e_acados_reset"(nlp_solver_capsule * capsule, int reset_qp_solver_mem, int reset_numerical_values, int reset_solver_options, int reset_x_to_x0_bar)
    int acados_free "ocp_safe_mpc_f5099c06c437_ee23731e_acados_free"(nlp_solver_capsule * capsule)
    void acados_print_stats "ocp_safe_mpc_f5099c06c437_ee23731e_acados_print_stats"(nlp_solver_capsule * capsule)

    int acados_custom_update "ocp_safe_mpc_f5099c06c437_ee23731e_acados_custom_update"(nlp_solver_capsule* capsule, double * data, int data_len)

    acados_solver_common.ocp_nlp_in *acados_get_nlp_in "ocp_safe_mpc_f5099c06c437_ee23731e_acados_get_nlp_in"(nlp_solver_capsule * capsule)
    acados_solver_common.ocp_nlp_out *acados_get_nlp_out "ocp_safe_mpc_f5099c06c437_ee23731e_acados_get_nlp_out"(nlp_solver_capsule * capsule)
    acados_solver_common.ocp_nlp_out *acados_get_sens_out "ocp_safe_mpc_f5099c06c437_ee23731e_acados_get_sens_out"(nlp_solver_capsule * capsule)
    acados_solver_common.ocp_nlp_solver *acados_get_nlp_solver "ocp_safe_mpc_f5099c06c437_ee23731e_acados_get_nlp_solver"(nlp_solver_capsule * capsule)
    acados_solver_common.ocp_nlp_config *acados_get_nlp_config "ocp_safe_mpc_f5099c06c437_ee23731e_acados_get_nlp_config"(nlp_solver_capsule * capsule)
    void *acados_get_nlp_opts "ocp_safe_mpc_f5099c06c437_ee23731e_acados_get_nlp_opts"(nlp_solver_capsule * capsule)
    acados_solver_common.ocp_nlp_dims *acados_get_nlp_dims "ocp_safe_mpc_f5099c06c437_ee23731e_acados_get_nlp_dims"(nlp_solver_capsule * capsule)
    acados_solver_common.ocp_nlp_plan *acados_get_nlp_plan "ocp_safe_mpc_f5099c06c437_ee23731e_acados_get_nlp_plan"(nlp_solver_capsule * capsule)
