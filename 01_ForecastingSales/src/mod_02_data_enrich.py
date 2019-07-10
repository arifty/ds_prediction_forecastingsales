# ------------------
# Find Correlations
# ------------------
prod_info_filt = prod_info[prod_info['Style_display_code'] \
                           .isin(ls_pos+ls_scoring+booking_missing_style)]
prod_info_dedup = prod_info_filt[['Style_display_code','age_desc', 'gender_desc',
                                  'product_family','category']].drop_duplicates()

cat_features = ['age_desc', 'gender_desc', 'product_family','category' ]


# duration: runs for 13 mins because of corr() function
corr_data = calculate_correlations(prod_info_dedup, 'Style_display_code', cat_features, 
                                   pos_missing+booking_missing)





# ----- POS_DATA ENRICH -------
# format the data before enriching correlated data
pos_missing_style = corr_data[(corr_data['order']=='0') & 
                                    (corr_data['Style_display_code'].isin(pos_missing))]

corr_pos_enriched = pos_missing_style.merge(pos_data \
                          .rename(columns={'Style_display_code': 'correlated_Style_display_code'}), 
                          how="inner", on='correlated_Style_display_code') \
                          .drop(['correlated_Style_display_code','order'], axis=1) \
                          .set_index('Style_display_code')

# enrich the pos_data with missing scoring style codes
pos_enriched = pos_data[pos_data['Style_display_code'].isin(ls_scoring)].set_index('Style_display_code') \
                .append(corr_pos_enriched)



# convert the date field to datetime type
pos_enriched['Activity_Date'] = pd.to_datetime(pos_enriched['Activity_Date'], 
                                               errors ='coerce')
# fetch the season and corresponding dates
season_weeks = pos_enriched.reset_index()[['SesnYrCd','Activity_Date']].drop_duplicates() \
                .sort_values(by='Activity_Date')

# calculate the week of season column by ranking 13 weeks
season_weeks['week_of_season'] = season_weeks.groupby('SesnYrCd')['Activity_Date'] \
                                  .rank(ascending=True)
# merge the new column with the pos_enrich
pos_enriched = pos_enriched.reset_index().merge(season_weeks, on=['SesnYrCd','Activity_Date'], 
                                 how='inner')

# Aggregate the data  
pos_enriched_agg = pos_enriched.fillna(0).groupby(groupby_sum_cols).sum().reset_index()

 

# ----- BOOKING ENRICH -------
# format the data before enriching correlated data
booking_missing_style = corr_data[(corr_data['order']=='0') & 
                                    (corr_data['Style_display_code'].isin(booking_missing))]

corr_booking_enriched = booking_missing_style.merge(bookings \
                          .rename(columns={'Style_display_code': 'correlated_Style_display_code'}), 
                          how="inner", on='correlated_Style_display_code') \
                          .drop(['correlated_Style_display_code','order'], axis=1) \
                          .set_index('Style_display_code')

# enrich the bookings data with missing scoring style codes
bookings_enriched = bookings[bookings['Style_display_code'].isin(ls_scoring)].set_index('Style_display_code') \
                      .append(corr_booking_enriched)


# find the not enriched booking data
ls_bookings_enriched = corr_booking_enriched.index.unique().tolist()
booking_not_enriched = list(set(booking_missing) - set(ls_bookings_enriched))




# ------- PAUSE -----------
# save the data
pickle.dump(bookings_enriched, open(staging_dir+ '/bookings_enriched_py3.p', 'wb'))
pickle.dump(pos_enriched, open(staging_dir+ '/pos_enriched_py3.p', 'wb'))

# read again from pickle
bookings_enriched = pickle.load(open(staging_dir+ '/bookings_enriched_py3.p', "rb" ) )
pos_enriched = pickle.load(open(staging_dir+ '/pos_enriched_py3.p', "rb" ) )
# ------- PAUSE -----------


# Re-enrich the still missing bookings data
bookings_agg_1 = bookings_enriched.reset_index() \
                  .fillna(0).groupby(groupby_sum_cols).sum().reset_index()
bookings_agg_2 = bookings_agg_1.groupby(groupby_mean_cols).mean().reset_index()

bookings_re_enriched = pd.concat([bookings_agg_2] * len(booking_not_enriched),
                                 keys=booking_not_enriched) \
                        .reset_index(level=1, drop=True) \
                        .rename_axis('Style_display_code').reset_index()

# aggregate bookings at Style code & Season level
bookings_enrich_agg = bookings_agg_1.append(bookings_re_enriched)




# Save enriched data
pickle.dump(bookings_enrich_agg, open(staging_dir+ '/bookings_enriched_agg.p', 'wb'))
pickle.dump(pos_enriched_agg, open(staging_dir+ '/pos_enriched_agg_v2.p', 'wb'))


# ------- START AFTER PAUSE -----------
bookings_enrich_agg = pickle.load(open(staging_dir+ '/bookings_enriched_agg.p', "rb" ))
pos_enriched_agg = pickle.load(open(staging_dir+ '/pos_enriched_agg_v2.p', "rb" ))
# ------- START AFTER PAUSE -----------

