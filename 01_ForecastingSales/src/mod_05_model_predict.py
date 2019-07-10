# load the functions for Outlier detection and Regression model
exec(open(os.path.join(code_dir, "fn_03_regression_algo.py")).read())

# Predict with future dataset
# ---------------------------

#pos_booking_totalenriched = pickle.load(open(output_dir+ '/pos_booking_enrich.p', "rb" ) )
# filter only SUMMER 2016 entries, which needs to be predicted
season_topredict = 'SU2016'
pos_booking_enrich = base_df[base_df["SesnYrCd"] == season_topredict] \
        .merge(bookings_enrich_agg[bookings_enrich_agg["SesnYrCd"] == season_topredict], 
                             on=groupby_booking_cols, how='left').fillna(0)

pos_booking_enrich['season'] = pos_booking_enrich['SesnYrCd'].str[:2]
pos_booking_enrich['year_id'] = pos_booking_enrich['SesnYrCd'].str[2:]
pos_booking_enrich.drop(['SesnYrCd'], axis=1, inplace=True)
pos_booking_enrich['NetSlsUnts_WTD'] = 0
  
# ---------------------------
# call the regression model
forecasted_results = model_regression_predict(pos_booking_enrich,ls_stylecodes,numerical_feat, 
                                              categorical_feat, model_name)

# save the outliers data for reference
pickle.dump(forecasted_results, open(output_dir+"/forecasted_results_"+model_name+".p", 'wb'))

# write to Excel 
writer = pd.ExcelWriter(output_dir+"/forecasted_results_"+model_name+".xlsx")
forecasted_results.sort_values(['Style_display_code']).to_excel(writer,'predictions', index = False)
writer.save()

