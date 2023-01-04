
def apply_user_defined_normalization_correction(df_norm_info):
    try:
        df_norm_info.loc['min', 'linear_vel_x'] = -df_norm_info.loc['max', 'linear_vel_x']
        df_norm_info.loc['min', 'translational_control'] = -df_norm_info.loc['max', 'translational_control']
    except KeyError:
        pass
    return df_norm_info