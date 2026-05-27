-- BUSINESS QUESTION: How does a delay in promotion (measured by years since last promotion relative to job level) influence attrition rates?
SELECT 
    CASE 
        WHEN years_since_last_promotion >= 5 THEN 'Severe Delay (5+ Years)'
        WHEN years_since_last_promotion >= 3 THEN 'Moderate Delay (3-4 Years)'
        ELSE 'Normal/Recent Promotion (<3 Years)'
    END as promotion_delay_status,
    COUNT(*) as total_employees,
    SUM(attrition) as departed_employees,
    ROUND(CAST(SUM(attrition) AS NUMERIC) / COUNT(*) * 100, 2) as attrition_rate_percent
FROM 
    employees
GROUP BY 
    CASE 
        WHEN years_since_last_promotion >= 5 THEN 'Severe Delay (5+ Years)'
        WHEN years_since_last_promotion >= 3 THEN 'Moderate Delay (3-4 Years)'
        ELSE 'Normal/Recent Promotion (<3 Years)'
    END
ORDER BY 
    attrition_rate_percent DESC;
