SELECT ItemId, ColorId, FROM Item;


SELECT i.ItemId,
		ia.ItemAttributeId,
		ia.ItemAttributeValueSetId,
		iavs.Name AS AttributeValueSetName,
		ia.Name AS AttributeName,
		iavt.AttributeValue,
		iavt.ItemAttributeValueId,
		iavt.ItemAttributeValueSetOptionId
FROM Item i 
JOIN ItemAttributeValue iav
	ON i.ItemId = iav.ItemId
	AND iav.ItemAttributeId >= 90000
	AND i.ItemId = 303529
JOIN ItemAttribute ia 
	ON ia.ItemAttributeId = iav.ItemAttributeId
JOIN ItemAttributeValueText iavt 
	ON iavt.LanguageId =1
	AND iavt.ItemAttributeValueId = iav.ItemAttributeValueId
LEFT JOIN ItemAttributeValueSet iavs
	ON iavs.ItemAttributeValueSetId = ia.ItemAttributeValueSetId
WHERE i.ItemId = 303529;




SELECT 
 i.ItemId
#,iav.ItemAttributeSetId
,ias.Name AS AttributeSetName
,ia.Name AS AttributeName
#,iav.ItemAttributeValueId
,COALESCE(iavt.AttributeValue, iavso.Name) AS AttributeValue
#,iavt.ItemAttributeValueSetOptionId
#,iav.ItemAttributeId 
FROM Item i
JOIN ItemAttributeValue iav
	ON i.ItemId = iav.ItemId
	AND iav.ItemAttributeSetId>=90000
JOIN ItemAttribute ia
	ON iav.ItemAttributeId = ia.ItemAttributeId 
JOIN ItemAttributeSet ias
	ON iav.ItemAttributeSetId = ias.ItemAttributeSetId 
JOIN ItemAttributeValueText iavt
	ON iav.ItemAttributeValueId  = iavt.ItemAttributeValueId
	AND iavt.LanguageId = 1 
LEFT JOIN ItemAttributeValueSetOption iavso
	ON iavso.ItemAttributeValueSetOptionId = iavt.ItemAttributeValueSetOptionId 
WHERE i.Status = 1
	AND (iavt.AttributeValue IS NOT NULL OR iavt.ItemAttributeValueSetOptionId IS NOT NULL);

SELECT itm.ItemId, m.MediaId, itm.`Order`,
CONCAT('https://www.connox.de/cdn/shop/files', REGEXP_SUBSTR(mt.File, '(?i)/[^/]+[.](jpg|jpeg|png|gif|webp|bmp|tif|tiff|svg|heic|avif)$')) AS PartialPath
FROM Item i
JOIN ItemToMedia itm
	ON itm.ItemId = i.ItemId 
	AND i.Status = 1
JOIN Media m
	ON itm.MediaId = m.MediaId
	AND (m.SalesDomainId = 100000 OR m.SalesDomainId = 100004)
JOIN MediaText mt 
	ON m.MediaId = mt.MediaId 
	AND mt.LanguageId = 1;


select * from Item 
where itemId in (100361,
100362,
288733,
289958,
290668,
306338,
311794,
330164,
397312,
416325,
417570);



SELECT ItemId, MIN(`Order`) FROM ItemToMedia
GROUP BY 1
LIMIT 50000;


SHOW TABLES;