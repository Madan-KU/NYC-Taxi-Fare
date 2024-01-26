# Dataset Columns Explanation

## Original Fields:
1. **request_datetime**: The date and time when the service was requested by the passenger.
2. **on_scene_datetime**: The date and time when the vehicle arrived on the scene after the service was requested.
3. **pickup_datetime**: The date and time when the passenger was picked up.
4. **dropoff_datetime**: The date and time when the passenger was dropped off at their destination.
5. **PULocationID**: TLC Taxi Zone ID where the passenger was picked up.
6. **DOLocationID**: TLC Taxi Zone ID where the passenger was dropped off.
7. **trip_miles**: The total distance of the trip in miles.
8. **trip_time**: The total time taken for the trip (probably in minutes).
9. **base_passenger_fare**: The basic fare for the trip, excluding any additional charges like tolls or surcharges.
10. **tolls**: The total amount of all tolls paid during the trip.
11. **bcf**: Base Charge Fare. It's an initial fixed charge in the taxi fare.
12. **sales_tax**: The amount of sales tax charged for the trip.
13. **congestion_surcharge**: A surcharge applied to trips passing through a designated congestion area, usually in busy parts of the city like Manhattan.
14. **airport_fee**: Additional fee charged for trips that are to or from the airport.
15. **tips**: The tip amount given by the passenger. This is usually higher for credit card payments and may be zero for cash payments if the passenger didn't tip.
16. **driver_pay**: The total amount paid to the driver. This would be the target variable if you're predicting the driver's earnings.

## Engineered Fields:
17. **duration_minutes**: Duration of the trip calculated as the difference between drop-off and pickup times in minutes.
18-21. **Datetime Decomposed Features**: For each of the datetime columns (`request_datetime`, `on_scene_datetime`, `pickup_datetime`, `dropoff_datetime`), we've extracted:
    - **Hour**: The hour component of the datetime.
    - **Day**: The day component of the datetime.
    - **Day of Week**: The day of the week (0=Monday, 6=Sunday).
    - **Month**: The month component of the datetime.
22. **wait_time_minutes**: Duration in minutes between when the service was requested and when the vehicle arrived on the scene.
23. **service_time_minutes**: Duration in minutes between when the vehicle arrived on the scene and when the passenger was dropped off.
24. **average_speed**: The average speed of the trip calculated as `trip_miles` divided by `trip_time`.
25. **fare_per_mile**: The fare charged per mile, calculated as `base_passenger_fare` divided by `trip_miles`.
26. **total_charges**: The sum of various charges like `base_passenger_fare`, `tolls`, `bcf`, `sales_tax`, `congestion_surcharge`, and `airport_fee`.
27. **tip_percentage**: The percentage of the tip with respect to the total charges.
28. **expected_pay_without_tips**: The expected pay to the driver excluding the tips, calculated as `driver_pay` minus `tips`.
29. **miles_times_speed**: An interaction feature calculated as `trip_miles` multiplied by `average_speed`.
