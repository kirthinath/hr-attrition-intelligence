-- BUSINESS QUESTION: How does attrition risk distribute across tenure groups, and who are the most vulnerable cohorts?
SELECT 
    tenure_group,
    COUNT(*) as total_employees,
    SUM(attrition) as departed_employees,
    ROUND(CAST(SUM(attrition) AS NUMERIC) / COUNT(*) * 100, 2) as attrition_rate_percent,
    ROUND(AVG(years_with_curr_manager), 2) as avg_years_with_manager
FROM 
    employees
GROUP BY 
    tenure_group
ORDER BY 
    attrition_rate_percent DESC;
