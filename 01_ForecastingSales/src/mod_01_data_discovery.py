# scoring list
scoring_list = pd.read_csv(input_dir+"/cs_scoring_list.csv")
scoring_list.head(6)

# pos data
pos_data = pd.read_csv(input_dir+"/cs_pos_data.csv")
pos_data.head(6)

# product data
prod_info = pd.read_csv(input_dir+"/cs_prod_info.csv")
prod_info.head(6)

# bookings data
bookings = pd.read_csv(input_dir+"/cs_bookings.csv")
bookings.head(6)


# get variables list separately
ls_scoring = scoring_list['Style_display_code'].tolist()
ls_pos = pos_data['Style_display_code'].unique().tolist()
ls_prod_info = prod_info['Style_display_code'].unique().tolist()
ls_booking = bookings['Style_display_code'].unique().tolist()

ls_terrirory = bookings['New_Territory'].unique().tolist()
ls_season = bookings['SesnYrCd'].unique().tolist()

# formulate the base dataframe with all possible variables
# this will be the baseline, on which training, testing and forecasting will be done
weeknr_list = list(range(1,14))
base_df = cartesian_lists_toDF (ls_scoring, ls_season, weeknr_list, ls_terrirory, 
                                groupby_sum_cols)


# Some Data validation steps
# --------------------------
# Check the missing style_codes
booking_missing_style = list(set(ls_scoring) - set(ls_booking))

# 839 codes missing
pos_missing = list(set(ls_scoring) - set(ls_pos))
len(pos_missing)

# 648 codes missing
booking_missing = list(set(ls_scoring) - set(ls_booking))
len(booking_missing)

# booking vs prod_info
prod_booking_missing = list(set(booking_missing) - set(ls_prod_info))
len(prod_booking_missing)
