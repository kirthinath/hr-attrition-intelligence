-- BUSINESS QUESTION: What are the overall historical attrition and retention rates for the organization, and what is the total estimated financial loss from attrition?
SELECT 
    COUNT(*) as total_historical_headcount,
    SUM(attrition) as total_departures,
    COUNT(*) - SUM(attrition) as current_active_headcount,
    ROUND((1.0 - CAST(SUM(attrition) AS NUMERIC) / COUNT(*)) * 100, 2) as retention_rate_percent,
    ROUND(CAST(SUM(attrition) AS NUMERIC) / COUNT(*) * 100, 2) as attrition_rate_percent,
    SUM(attrition * monthly_income * 12 * 1.5) as total_estimated_cost_loss_usd
FROM 
    employees;
