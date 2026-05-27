-- BUSINESS QUESTION: What is the impact of working overtime on employee attrition, and how does this correlate with self-reported work-life balance?
SELECT 
    overtime,
    work_life_balance,
    COUNT(*) as total_employees,
    SUM(attrition) as departed_employees,
    ROUND(CAST(SUM(attrition) AS NUMERIC) / COUNT(*) * 100, 2) as attrition_rate_percent
FROM 
    employees
GROUP BY 
    overtime, 
    work_life_balance
ORDER BY 
    overtime DESC, 
    work_life_balance ASC;
