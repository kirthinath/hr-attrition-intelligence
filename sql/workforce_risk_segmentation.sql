-- BUSINESS QUESTION: How is the active workforce segmented across strategic HR risk and engagement cohorts?
SELECT 
    primary_segment,
    COUNT(*) as active_employee_count,
    ROUND(AVG(attrition_probability) * 100, 2) as avg_attrition_probability_percent,
    ROUND(AVG(monthly_income), 2) as avg_monthly_income_usd
FROM 
    employees
WHERE 
    attrition = 0
GROUP BY 
    primary_segment
ORDER BY 
    active_employee_count DESC;
