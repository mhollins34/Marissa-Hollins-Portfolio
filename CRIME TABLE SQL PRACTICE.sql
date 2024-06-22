SELECT description, COUNT(*) AS new_description
FROM crime_table
GROUP BY description
ORDER BY new_description DESC;

SELECT * from crime_table;

SELECT ward, COUNT(*) AS Number_of_Wards
FROM crime_table
GROUP BY ward
ORDER BY Number_of_Wards DESC;

SELECT year, COUNT(arrest) AS arrests_made
FROM crime_table
WHERE arrest::bool=TRUE
GROUP BY year
ORDER BY year DESC;

SELECT year, COUNT(arrest) AS arrests_made
FROM crime_table
WHERE arrest::bool=FALSE
GROUP BY year
ORDER BY arrests_made ASC;
