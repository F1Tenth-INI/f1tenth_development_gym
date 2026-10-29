/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * Licensed under the 2-Clause BSD License.
 */


#ifndef safe_mpc_5293571f4164_COST
#define safe_mpc_5293571f4164_COST

#ifdef __cplusplus
extern "C" {
#endif


// Cost at initial shooting node

int safe_mpc_5293571f4164_cost_ext_cost_0_fun(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int safe_mpc_5293571f4164_cost_ext_cost_0_fun_work(int *, int *, int *, int *);
const int *safe_mpc_5293571f4164_cost_ext_cost_0_fun_sparsity_in(int);
const int *safe_mpc_5293571f4164_cost_ext_cost_0_fun_sparsity_out(int);
int safe_mpc_5293571f4164_cost_ext_cost_0_fun_n_in(void);
int safe_mpc_5293571f4164_cost_ext_cost_0_fun_n_out(void);

int safe_mpc_5293571f4164_cost_ext_cost_0_fun_jac_hess(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int safe_mpc_5293571f4164_cost_ext_cost_0_fun_jac_hess_work(int *, int *, int *, int *);
const int *safe_mpc_5293571f4164_cost_ext_cost_0_fun_jac_hess_sparsity_in(int);
const int *safe_mpc_5293571f4164_cost_ext_cost_0_fun_jac_hess_sparsity_out(int);
int safe_mpc_5293571f4164_cost_ext_cost_0_fun_jac_hess_n_in(void);
int safe_mpc_5293571f4164_cost_ext_cost_0_fun_jac_hess_n_out(void);

int safe_mpc_5293571f4164_cost_ext_cost_0_fun_jac(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int safe_mpc_5293571f4164_cost_ext_cost_0_fun_jac_work(int *, int *, int *, int *);
const int *safe_mpc_5293571f4164_cost_ext_cost_0_fun_jac_sparsity_in(int);
const int *safe_mpc_5293571f4164_cost_ext_cost_0_fun_jac_sparsity_out(int);
int safe_mpc_5293571f4164_cost_ext_cost_0_fun_jac_n_in(void);
int safe_mpc_5293571f4164_cost_ext_cost_0_fun_jac_n_out(void);









// Cost at path shooting node

int safe_mpc_5293571f4164_cost_ext_cost_fun(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int safe_mpc_5293571f4164_cost_ext_cost_fun_work(int *, int *, int *, int *);
const int *safe_mpc_5293571f4164_cost_ext_cost_fun_sparsity_in(int);
const int *safe_mpc_5293571f4164_cost_ext_cost_fun_sparsity_out(int);
int safe_mpc_5293571f4164_cost_ext_cost_fun_n_in(void);
int safe_mpc_5293571f4164_cost_ext_cost_fun_n_out(void);

int safe_mpc_5293571f4164_cost_ext_cost_fun_jac_hess(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int safe_mpc_5293571f4164_cost_ext_cost_fun_jac_hess_work(int *, int *, int *, int *);
const int *safe_mpc_5293571f4164_cost_ext_cost_fun_jac_hess_sparsity_in(int);
const int *safe_mpc_5293571f4164_cost_ext_cost_fun_jac_hess_sparsity_out(int);
int safe_mpc_5293571f4164_cost_ext_cost_fun_jac_hess_n_in(void);
int safe_mpc_5293571f4164_cost_ext_cost_fun_jac_hess_n_out(void);

int safe_mpc_5293571f4164_cost_ext_cost_fun_jac(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int safe_mpc_5293571f4164_cost_ext_cost_fun_jac_work(int *, int *, int *, int *);
const int *safe_mpc_5293571f4164_cost_ext_cost_fun_jac_sparsity_in(int);
const int *safe_mpc_5293571f4164_cost_ext_cost_fun_jac_sparsity_out(int);
int safe_mpc_5293571f4164_cost_ext_cost_fun_jac_n_in(void);
int safe_mpc_5293571f4164_cost_ext_cost_fun_jac_n_out(void);








// Cost at terminal shooting node

int safe_mpc_5293571f4164_cost_y_e_fun(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int safe_mpc_5293571f4164_cost_y_e_fun_work(int *, int *, int *, int *);
const int *safe_mpc_5293571f4164_cost_y_e_fun_sparsity_in(int);
const int *safe_mpc_5293571f4164_cost_y_e_fun_sparsity_out(int);
int safe_mpc_5293571f4164_cost_y_e_fun_n_in(void);
int safe_mpc_5293571f4164_cost_y_e_fun_n_out(void);

int safe_mpc_5293571f4164_cost_y_e_fun_jac_ut_xt(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int safe_mpc_5293571f4164_cost_y_e_fun_jac_ut_xt_work(int *, int *, int *, int *);
const int *safe_mpc_5293571f4164_cost_y_e_fun_jac_ut_xt_sparsity_in(int);
const int *safe_mpc_5293571f4164_cost_y_e_fun_jac_ut_xt_sparsity_out(int);
int safe_mpc_5293571f4164_cost_y_e_fun_jac_ut_xt_n_in(void);
int safe_mpc_5293571f4164_cost_y_e_fun_jac_ut_xt_n_out(void);
int safe_mpc_5293571f4164_cost_y_e_hess(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int safe_mpc_5293571f4164_cost_y_e_hess_work(int *, int *, int *, int *);
const int *safe_mpc_5293571f4164_cost_y_e_hess_sparsity_in(int);
const int *safe_mpc_5293571f4164_cost_y_e_hess_sparsity_out(int);
int safe_mpc_5293571f4164_cost_y_e_hess_n_in(void);
int safe_mpc_5293571f4164_cost_y_e_hess_n_out(void);




#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // safe_mpc_5293571f4164_COST
