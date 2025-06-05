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

    """Neo4j queries for exploring fashion concepts and aesthetics"""

    def get_aesthetic_concepts(self, aesthetic_name: str) -> str:
        """Get all concepts related to a specific aesthetic"""
        return f"""
        MATCH (e:Эстетика {{name: "{aesthetic_name}"}})<-[:ОТНОСИТСЯ_К]-(c:Концепт)
        RETURN c.name as concept, labels(c) as labels
        """

    def get_aesthetic_combinations(self, aesthetic_name: str) -> str:
        """Get combinations between items within an aesthetic"""
        return f"""
        MATCH (e:Эстетика {{name: "{aesthetic_name}"}})<-[:ОТНОСИТСЯ_К]-(c1:Концепт)
        MATCH (c1)-[r:СОЧЕТАЕТСЯ_С]-(c2:Концепт)
        WHERE (c2)-[:ОТНОСИТСЯ_К]->(:Эстетика {{name: "{aesthetic_name}"}})
        RETURN c1.name as item1, type(r) as relation, c2.name as item2
        """

    def get_aesthetic_materials(self, aesthetic_name: str) -> str:
        """Get materials commonly used in an aesthetic"""
        return f"""
        MATCH (e:Эстетика {{name: "{aesthetic_name}"}})<-[:ОТНОСИТСЯ_К]-(c:Концепт)
        MATCH (c)-[:СДЕЛАН_ИЗ]->(m:Материал)
        RETURN c.name as item, collect(m.name) as materials
        """

    def get_aesthetic_clothing(self, aesthetic_name: str) -> str:
        """Get clothing items in an aesthetic"""
        return f"""
        MATCH (e:Эстетика {{name: "{aesthetic_name}"}})<-[:ОТНОСИТСЯ_К]-(c:Концепт)
        MATCH (c)-[:ЯВЛЯЕТСЯ_ОДЕЖДОЙ]->(o:Одежда)
        RETURN c.name as concept, o.name as clothing_type
        """

    def get_aesthetic_shoes(self, aesthetic_name: str) -> str:
        """Get shoes in an aesthetic"""
        return f"""
        MATCH (e:Эстетика {{name: "{aesthetic_name}"}})<-[:ОТНОСИТСЯ_К]-(c:Концепт)
        MATCH (c)-[:ЯВЛЯЕТСЯ_ОБУВЬЮ]->(s:Обувь)
        RETURN c.name as concept, s.name as shoe_type
        """

    def get_cross_aesthetic_combinations(self, aesthetic_name: str) -> str:
        """Get combinations between this aesthetic and others"""
        return f"""
        MATCH (e1:Эстетика {{name: "{aesthetic_name}"}})<-[:ОТНОСИТСЯ_К]-(c1:Концепт)
        MATCH (c1)-[r:СОЧЕТАЕТСЯ_С]-(c2:Концепт)-[:ОТНОСИТСЯ_К]->(e2:Эстетика)
        WHERE e1 <> e2
        RETURN c1.name as from_item, c2.name as to_item, e2.name as other_aesthetic
        """

    def find_concept_combinations(self, concept_name: str) -> str:
        """Find all combinations for a specific concept, considering bidirectional СОЧЕТАЕТСЯ_С relationships"""
        return f"""
        MATCH (c1:Концепт {{name: "{concept_name}"}})-[r:СОЧЕТАЕТСЯ_С]-(c2:Концепт)
        OPTIONAL MATCH (c2)-[:ОТНОСИТСЯ_К]->(e:Эстетика)
        RETURN c2.name as matching_item, collect(e.name) as aesthetics
        """

    def find_related_concepts(self, text: str) -> str:
        """Find concepts and their relationships based on text input"""
        return """
        WITH $text as input
        MATCH (c:Концепт)
        WHERE c.name CONTAINS input OR any(alias IN c.aliases WHERE alias CONTAINS input)
        WITH collect(c) as concepts
        UNWIND concepts as c1
        OPTIONAL MATCH (c1)-[r]-(c2:Концепт)
        WHERE c2 IN concepts
        RETURN c1.name as concept1, type(r) as relation, c2.name as concept2
        """

    def get_concept_full_info(self, concept_name: str) -> str:
        """Get complete information about a concept including its aesthetics, materials, and combinations"""
        return f"""
        MATCH (c:Концепт {{name: "{concept_name}"}})
        OPTIONAL MATCH (c)-[:ОТНОСИТСЯ_К]->(e:Эстетика)
        OPTIONAL MATCH (c)-[:СДЕЛАН_ИЗ]->(m:Материал)
        OPTIONAL MATCH (c)-[r:СОЧЕТАЕТСЯ_С]-(other:Концепт)
        RETURN c.name as concept,
               collect(DISTINCT e.name) as aesthetics,
               collect(DISTINCT m.name) as materials,
               collect(DISTINCT {{item: other.name, relation: type(r)}}) as combinations
        """ 