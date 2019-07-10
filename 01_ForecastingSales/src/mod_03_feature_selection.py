
# Prepare the features
# ---------------------
# merge bookings to pos data
pos_booking_data = pos_enriched_agg.merge(bookings_enrich_agg, how="left", on=groupby_booking_cols) \
                    .fillna(0)

  #.drop('OHInvUnts_WTD', axis=1)


# data quality fixes
# ------------------
# update the sales data, whenever it is more than the bookings value
# Cap it to the bookings value for those cases
pos_booking_data.loc[((pos_booking_data['bookings']>0) & 
                      (pos_booking_data['NetSlsUnts_WTD']>pos_booking_data['bookings'])), 'NetSlsUnts_WTD'] \
            =  pos_booking_data.loc[((pos_booking_data['bookings']>0) & 
                                     (pos_booking_data['NetSlsUnts_WTD']>pos_booking_data['bookings'])), 'bookings']


# flag too few samples as difficult to predict, n_samples should be 3 or more
low_observations = pos_booking_data.groupby('Style_display_code')['Style_display_code'] \
                    .filter(lambda x: x.count() < 3)
pos_booking_data['low_obsv'] = low_observations

# transform the low observation column to 0 = if no low observation and 1 = if low observation
pos_booking_data['low_obsv_flag'] = np.where((pos_booking_data['low_obsv'].isnull()), 0, 1)

# Fill all missing data with mean for style codes with missing lines from base_df
#pos_booking_totalenriched = base_df.merge(pos_booking_data,on=groupby_sum_cols, how='left').fillna(0)
pos_booking_totalenriched = pos_booking_data[pos_booking_data['low_obsv_flag']==0]

# split the season and year fields
pos_booking_totalenriched['season'] = pos_booking_totalenriched['SesnYrCd'].str[:2]
pos_booking_totalenriched['year_id'] = pos_booking_totalenriched['SesnYrCd'].str[2:]
pos_booking_totalenriched.drop(['SesnYrCd','low_obsv'], axis=1, inplace=True)

# ------- SAVE FOR FUTURE USE -----------
pickle.dump(pos_booking_totalenriched, open(output_dir+ '/pos_booking_enrich.p', 'wb'))

# read again from pickle
pos_booking_totalenriched = pickle.load(open(output_dir+ '/pos_booking_enrich.p', "rb" ))
# ------- SAVE FOR FUTURE USE -----------



