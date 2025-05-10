class Neo4jQueries:
    # Schema and Structure
    GET_GRAPH_STRUCTURE = """
        CALL db.schema.visualization()
        YIELD nodes, relationships
        RETURN nodes, relationships
    """

    # Node Operations
    CREATE_NODE = """
        CREATE (n:$label $properties)
        RETURN id(n) as node_id
    """

    # Relationship Operations
    CREATE_RELATIONSHIP = """
        MATCH (a:$from_label), (b:$to_label)
        WHERE a.name = $from_name AND b.name = $to_name
        CREATE (a)-[r:$relationship_type $props]->(b)
        RETURN id(r) as rel_id
    """

    # Entity Extraction
    EXTRACT_ENTITIES = """
        MATCH (n)
        WHERE n.name CONTAINS $prompt
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN n, r, m
    """

    # Style Recommendations
    GET_STYLE_RECOMMENDATIONS = """
        MATCH (s:Style {name: $style})-[r]->(n)
        RETURN n.name as item, type(r) as relationship
    """

    # Fashion-specific queries
    GET_STYLE_ITEMS = """
        MATCH (s:Style {name: $style})-[r:INCLUDES]->(i:Item)
        RETURN i.name as item, r.season as season
    """

    GET_COLOR_COMBINATIONS = """
        MATCH (c1:Color {name: $color})-[r:COMBINES_WITH]->(c2:Color)
        RETURN c2.name as complementary_color, r.rating as compatibility
    """

    GET_EVENT_RECOMMENDATIONS = """
        MATCH (e:Event {name: $event})<-[:SUITABLE_FOR]-(s:Style)
        RETURN s.name as style, s.description as description
    """ 