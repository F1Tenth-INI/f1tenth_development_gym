
def apply_user_defined_normalization_correction(df_norm_info):

    df_norm_info.loc['min', 'linear_vel_x'] = -df_norm_info.loc['max', 'linear_vel_x']
    df_norm_info.loc['min', 'speed'] = -df_norm_info.loc['max', 'speed']

    return df_norm_info