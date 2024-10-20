CREATE OR REPLACE PROCEDURE `your_project.your_dataset.insert_transformed_data`()
BEGIN
  -- Declare variables for the sample and target table names
  DECLARE sample_table STRING DEFAULT 'your_project.your_dataset.sample_table';
  DECLARE target_table STRING DEFAULT 'your_project.your_dataset.transformed_table';

  -- Create the target table if it doesn't exist
  EXECUTE IMMEDIATE FORMAT("""
    CREATE TABLE IF NOT EXISTS `%s` (
      id INT64,
      transformed_text STRING,
      normalized_value FLOAT64,
      date_category STRING,
      is_valid BOOLEAN
    )
  """, target_table);

  -- Insert transformed data into the target table
  EXECUTE IMMEDIATE FORMAT("""
    INSERT INTO `%s` (id, transformed_text, normalized_value, date_category, is_valid)
    SELECT
      id,
      UPPER(text_column) AS transformed_text,
      (numeric_column - (SELECT AVG(numeric_column) FROM `%s`)) / (SELECT STDDEV(numeric_column) FROM `%s`) AS normalized_value,
      CASE
        WHEN date_column < DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) THEN 'Old'
        WHEN date_column BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) AND CURRENT_DATE() THEN 'Recent'
        ELSE 'Future'
      END AS date_category,
      REGEXP_CONTAINS(email_column, r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$') AS is_valid
    FROM `%s`
  """, target_table, sample_table, sample_table, sample_table);

  -- Log the number of rows inserted
  SELECT FORMAT("Inserted %d rows into %s", @@row_count, target_table);
END;
