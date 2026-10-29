/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * Licensed under the 2-Clause BSD License.
 */

#ifndef safe_mpc_f5099c06c437_MODEL
#define safe_mpc_f5099c06c437_MODEL

#ifdef __cplusplus
extern "C" {
#endif




int safe_mpc_f5099c06c437_dyn_disc_phi_fun(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int safe_mpc_f5099c06c437_dyn_disc_phi_fun_work(int *, int *, int *, int *);
const int *safe_mpc_f5099c06c437_dyn_disc_phi_fun_sparsity_in(int);
const int *safe_mpc_f5099c06c437_dyn_disc_phi_fun_sparsity_out(int);
int safe_mpc_f5099c06c437_dyn_disc_phi_fun_n_in(void);
int safe_mpc_f5099c06c437_dyn_disc_phi_fun_n_out(void);

int safe_mpc_f5099c06c437_dyn_disc_phi_fun_jac(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int safe_mpc_f5099c06c437_dyn_disc_phi_fun_jac_work(int *, int *, int *, int *);
const int *safe_mpc_f5099c06c437_dyn_disc_phi_fun_jac_sparsity_in(int);
const int *safe_mpc_f5099c06c437_dyn_disc_phi_fun_jac_sparsity_out(int);
int safe_mpc_f5099c06c437_dyn_disc_phi_fun_jac_n_in(void);
int safe_mpc_f5099c06c437_dyn_disc_phi_fun_jac_n_out(void);







int safe_mpc_f5099c06c437_dyn_disc_phi_fun_jac_hess(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int safe_mpc_f5099c06c437_dyn_disc_phi_fun_jac_hess_work(int *, int *, int *, int *);
const int *safe_mpc_f5099c06c437_dyn_disc_phi_fun_jac_hess_sparsity_in(int);
const int *safe_mpc_f5099c06c437_dyn_disc_phi_fun_jac_hess_sparsity_out(int);
int safe_mpc_f5099c06c437_dyn_disc_phi_fun_jac_hess_n_in(void);
int safe_mpc_f5099c06c437_dyn_disc_phi_fun_jac_hess_n_out(void);





#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // safe_mpc_f5099c06c437_MODEL
