-- BUSINESS QUESTION: How does attrition rate vary across different income bands, and what is the distribution of high-risk active employees across these bands?
SELECT 
    income_band,
    COUNT(*) as total_employees,
    SUM(attrition) as departed_employees,
    ROUND(CAST(SUM(attrition) AS NUMERIC) / COUNT(*) * 100, 2) as attrition_rate_percent,
    SUM(CASE WHEN attrition = 0 AND risk_level = 'High' THEN 1 ELSE 0 END) as active_high_risk_count
FROM 
    employees
GROUP BY 
    income_band
ORDER BY 
    MIN(monthly_income) ASC;
