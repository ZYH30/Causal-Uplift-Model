WITH candidate_events AS (
  SELECT
    c.stay_id,
    ce.charttime,
    CASE
      WHEN ce.itemid IN (220045) THEN 'heart_rate'
      WHEN ce.itemid IN (220050, 220179) THEN 'sbp'
      WHEN ce.itemid IN (220051, 220180) THEN 'dbp'
      WHEN ce.itemid IN (220052, 220181) THEN 'map'
      WHEN ce.itemid IN (220210) THEN 'resp_rate'
      WHEN ce.itemid IN (220277) THEN 'spo2'
      WHEN ce.itemid IN (223761, 223762) THEN 'temp_c'
      WHEN ce.itemid IN (220074) THEN 'cvp'
      WHEN ce.itemid IN (223835) THEN 'fio2'
      WHEN ce.itemid IN (223834, 227287, 227582) THEN 'o2_flow'
      WHEN ce.itemid IN (220739) THEN 'gcs_eye'
      WHEN ce.itemid IN (223900) THEN 'gcs_verbal'
      WHEN ce.itemid IN (223901) THEN 'gcs_motor'
      WHEN ce.itemid IN (226755, 227013) THEN 'gcs_total'
      WHEN ce.itemid IN (228096) THEN 'rass'
      WHEN ce.itemid IN (224639, 226512) THEN 'weight_kg'
      WHEN ce.itemid IN (220339) THEN 'peep'
      WHEN ce.itemid IN (224685, 224686) THEN 'tidal_volume'
      WHEN ce.itemid IN (224695) THEN 'peak_pressure'
    END AS feature,
    CASE
      WHEN ce.itemid IN (223761) THEN (ce.valuenum - 32.0) * 5.0 / 9.0
      WHEN ce.itemid IN (223835) AND ce.valuenum <= 1.5 THEN ce.valuenum * 100.0
      WHEN ce.itemid IN (220739, 223900, 223901, 228096) THEN COALESCE(
        ce.valuenum,
        SAFE_CAST(REGEXP_EXTRACT(ce.value, r'-?[0-9]+') AS FLOAT64)
      )
      ELSE ce.valuenum
    END AS numeric_value
  FROM `physionet-data.mimiciv_3_1_icu.chartevents` AS ce
  JOIN `YOUR_GCP_PROJECT.tlcd_chartevents_tmp.rbc_cohort_TIMESTAMP` AS c
    USING (stay_id)
  WHERE ce.itemid IN (220045, 220050, 220051, 220052, 220074, 220179, 220180, 220181, 220210, 220277, 220339, 220739, 223761, 223762, 223834, 223835, 223900, 223901, 224639, 224685, 224686, 224695, 226512, 226755, 227013, 227287, 227582, 228096)
    AND ce.charttime >= DATETIME_SUB(c.trigger_time, INTERVAL 24 HOUR)
    AND ce.charttime < c.trigger_time
), valid_events AS (
  SELECT *
  FROM candidate_events
  WHERE numeric_value IS NOT NULL
    AND (
      (feature = 'heart_rate' AND numeric_value BETWEEN 20.0 AND 250.0)
      OR (feature = 'sbp' AND numeric_value BETWEEN 30.0 AND 300.0)
      OR (feature = 'dbp' AND numeric_value BETWEEN 10.0 AND 200.0)
      OR (feature = 'map' AND numeric_value BETWEEN 20.0 AND 200.0)
      OR (feature = 'resp_rate' AND numeric_value BETWEEN 2.0 AND 80.0)
      OR (feature = 'spo2' AND numeric_value BETWEEN 50.0 AND 100.0)
      OR (feature = 'temp_c' AND numeric_value BETWEEN 30.0 AND 43.0)
      OR (feature = 'cvp' AND numeric_value BETWEEN -20.0 AND 50.0)
      OR (feature = 'fio2' AND numeric_value BETWEEN 21.0 AND 100.0)
      OR (feature = 'o2_flow' AND numeric_value BETWEEN 0.0 AND 100.0)
      OR (feature = 'gcs_eye' AND numeric_value BETWEEN 1.0 AND 4.0)
      OR (feature = 'gcs_verbal' AND numeric_value BETWEEN 1.0 AND 5.0)
      OR (feature = 'gcs_motor' AND numeric_value BETWEEN 1.0 AND 6.0)
      OR (feature = 'gcs_total' AND numeric_value BETWEEN 3.0 AND 15.0)
      OR (feature = 'rass' AND numeric_value BETWEEN -5.0 AND 4.0)
      OR (feature = 'weight_kg' AND numeric_value BETWEEN 20.0 AND 400.0)
      OR (feature = 'peep' AND numeric_value BETWEEN 0.0 AND 40.0)
      OR (feature = 'tidal_volume' AND numeric_value BETWEEN 50.0 AND 2000.0)
      OR (feature = 'peak_pressure' AND numeric_value BETWEEN 0.0 AND 80.0)
    )
), aggregate AS (
  SELECT
    stay_id,
    feature,
    ARRAY_AGG(numeric_value ORDER BY charttime DESC LIMIT 1)[OFFSET(0)] AS last,
    ARRAY_AGG(numeric_value ORDER BY charttime ASC LIMIT 1)[OFFSET(0)] AS first,
    MIN(numeric_value) AS min,
    MAX(numeric_value) AS max,
    AVG(numeric_value) AS mean,
    STDDEV_POP(numeric_value) AS sd,
    SAFE_DIVIDE(
      COVAR_POP(
        numeric_value,
        DATETIME_DIFF(charttime, DATETIME '2000-01-01 00:00:00', SECOND)
      ),
      VAR_POP(DATETIME_DIFF(charttime, DATETIME '2000-01-01 00:00:00', SECOND))
    ) * 3600.0 AS slope_per_hour,
    COUNT(*) AS count,
    COUNTIF(feature = 'heart_rate' AND numeric_value > 100) / COUNT(*) AS high_fraction,
    COUNTIF(
      (feature = 'map' AND numeric_value < 65)
      OR (feature = 'sbp' AND numeric_value < 90)
      OR (feature = 'spo2' AND numeric_value < 92)
    ) / COUNT(*) AS low_fraction
  FROM valid_events
  GROUP BY stay_id, feature
)
SELECT *
FROM aggregate
ORDER BY stay_id, feature
