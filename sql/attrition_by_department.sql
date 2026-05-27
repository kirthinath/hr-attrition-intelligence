-- BUSINESS QUESTION: What is the attrition rate across different departments, and what is the associated financial replacement cost for each?
SELECT 
    department,
    COUNT(*) as total_employees,
    SUM(attrition) as departed_employees,
    ROUND(CAST(SUM(attrition) AS NUMERIC) / COUNT(*) * 100, 2) as attrition_rate_percent,
    SUM(attrition * monthly_income * 12 * 1.5) as estimated_attrition_cost_usd
FROM 
    employees
GROUP BY 
    department
ORDER BY 
    attrition_rate_percent DESC;
