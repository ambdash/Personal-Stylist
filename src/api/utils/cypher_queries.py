class CypherQueries:
    # Item queries
    GET_ITEM_BY_NAME = """
    MATCH (i:FashionItem {name: $name})
    RETURN i
    """
    
    GET_ITEMS_BY_STYLE = """
    MATCH (i:FashionItem)-[:HAS_STYLE]->(s:Style {name: $style})
    RETURN i
    """
    
    GET_COMPATIBLE_ITEMS = """
    MATCH (i1:FashionItem {name: $item_name})-[r:COMPATIBLE_WITH]->(i2:FashionItem)
    WHERE r.score >= $min_score
    RETURN i2.name, i2.category, i2.color, r.score
    ORDER BY r.score DESC
    """
    
    # Outfit queries
    CREATE_OUTFIT = """
    CREATE (o:Outfit {
        id: $outfit_id,
        style: $style,
        occasion: $occasion,
        created_at: datetime()
    })
    WITH o
    UNWIND $items as item_name
    MATCH (i:FashionItem {name: item_name})
    CREATE (o)-[:CONTAINS]->(i)
    RETURN o
    """
    
    GET_OUTFIT_SUGGESTIONS = """
    MATCH (i1:FashionItem)-[r1:COMPATIBLE_WITH]->(i2:FashionItem)
    WHERE i1.style = $style AND i2.style = $style
    AND i1.category <> i2.category
    AND r1.score >= 0.8
    RETURN i1.name as item1, i2.name as item2, r1.score as compatibility
    ORDER BY r1.score DESC
    LIMIT 5
    """ 