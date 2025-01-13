class CarEquations:
    def __init__(self, lib):
        self.lib = lib

    def ks_model(self,
                 psi,  # Heading angle
                 v_x,  # Derivatives of state
                 delta,  # Steering angle
                 v_x_dot_cmd, delta_dot_cmd,  # Control inputs
                 l_wb,
                 ):
        s_x_dot = v_x * self.lib.cos(psi)
        s_y_dot = v_x * self.lib.sin(psi)
        delta_dot = delta_dot_cmd
        v_x_dot = v_x_dot_cmd
        psi_dot = (v_x / l_wb) * self.lib.tan(delta)
        psi_dot_dot = (v_x_dot * self.lib.tan(delta) / l_wb) \
                      + v_x * delta_dot / (l_wb * self.lib.cos(delta) ** 2)

        return (
            s_x_dot, s_y_dot,  # Rate of change of position
            psi_dot,  # Rate of change of heading angle

            v_x_dot,  # Rate of change of velocity
            psi_dot_dot,  # Rate of change of angular velocity

            delta_dot,  # Rate of change of steering angle
        )

    def pacejka_model(
            self,
            psi, v_x, v_y, psi_dot, delta, delta_dot, v_x_dot,  # Car state
            mu, lf, lr, h_cg, m, I_z, g_,  # Car parameters
            B_f, C_f, D_f, E_f, B_r, C_r, D_r, E_r,  # Pacejka parameters
    ):
        v_x = self.lib.where(v_x == 0, self.lib.constant(1e-8, self.lib.float32), v_x)
        alpha_f = -self.lib.atan((v_y + psi_dot * lf) / v_x) + delta
        alpha_r = -self.lib.atan((v_y - psi_dot * lr) / v_x)

        # compute vertical tire forces
        F_zf = m * (-v_x_dot * h_cg + g_ * lr) / (lr + lf)
        F_zr = m * (v_x_dot * h_cg + g_ * lf) / (lr + lf)

        F_yf = mu * F_zf * D_f * self.lib.sin(
            C_f * self.lib.atan(B_f * alpha_f - E_f * (B_f * alpha_f - self.lib.atan(B_f * alpha_f))))
        F_yr = mu * F_zr * D_r * self.lib.sin(
            C_r * self.lib.atan(B_r * alpha_r - E_r * (B_r * alpha_r - self.lib.atan(B_r * alpha_r))))

        d_pos_x = v_x * self.lib.cos(psi) - v_y * self.lib.sin(psi)
        d_pos_y = v_x * self.lib.sin(psi) + v_y * self.lib.cos(psi)
        d_psi = psi_dot
        d_v_x = v_x_dot
        d_v_y = 1 / m * (F_yr + F_yf) - v_x * psi_dot
        # print d_v_y + v_x * psi_dot
        # Should be equal to IMUs a_y

        d_psi_dot = 1 / I_z * (-lr * F_yr + lf * F_yf)

        return d_pos_x, d_pos_y, d_psi, d_v_x, d_v_y, d_psi_dot, delta_dot
