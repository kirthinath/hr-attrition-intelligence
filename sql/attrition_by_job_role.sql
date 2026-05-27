-- BUSINESS QUESTION: Which job roles suffer from the highest attrition rates, and what are their corresponding average satisfaction scores?
SELECT 
    job_role,
    COUNT(*) as total_employees,
    SUM(attrition) as departed_employees,
    ROUND(CAST(SUM(attrition) AS NUMERIC) / COUNT(*) * 100, 2) as attrition_rate_percent,
    ROUND(AVG(satisfaction_composite_score), 2) as avg_satisfaction_score
FROM 
    employees
GROUP BY 
    job_role
ORDER BY 
    attrition_rate_percent DESC;
