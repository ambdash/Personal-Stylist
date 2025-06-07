#!/usr/bin/env python3

from neo4j import GraphDatabase
import logging
from collections import defaultdict
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Neo4j connection settings
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"

def analyze_node_relations():
    """Analyze incoming relations for each node type (except Концепт)"""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            # Get all node types except Концепт
            node_types_query = """
            MATCH (n)
            WHERE NOT 'Концепт' IN labels(n)
            RETURN DISTINCT labels(n) as labels
            """
            
            node_types_result = session.run(node_types_query).data()
            all_node_types = set()
            for result in node_types_result:
                for label in result['labels']:
                    if label != 'Концепт':
                        all_node_types.add(label)
            
            logger.info(f"Found node types: {sorted(all_node_types)}")
            
            # Analyze each node type
            analysis_results = []
            
            for node_type in sorted(all_node_types):
                logger.info(f"\nAnalyzing {node_type} nodes...")
                
                # Get all nodes of this type with their incoming relation counts
                query = f"""
                MATCH (n:{node_type})
                OPTIONAL MATCH (source)-[r]->(n)
                WITH n, count(r) as incoming_relations, collect({{
                    source_type: labels(source)[0],
                    relation_type: type(r),
                    source_name: source.name
                }}) as relation_details
                RETURN n.name as name,
                       n.id as id,
                       incoming_relations,
                       relation_details
                ORDER BY incoming_relations DESC, n.name
                """
                
                results = session.run(query).data()
                
                # Statistics for this node type
                total_nodes = len(results)
                nodes_with_relations = sum(1 for r in results if r['incoming_relations'] > 0)
                nodes_without_relations = total_nodes - nodes_with_relations
                
                if results:
                    max_relations = max(r['incoming_relations'] for r in results)
                    avg_relations = sum(r['incoming_relations'] for r in results) / total_nodes
                else:
                    max_relations = 0
                    avg_relations = 0
                
                print(f"\n{node_type} ANALYSIS:")
                print(f"  Total nodes: {total_nodes}")
                print(f"  Nodes with relations: {nodes_with_relations}")
                print(f"  Nodes without relations: {nodes_without_relations}")
                print(f"  Max relations per node: {max_relations}")
                print(f"  Average relations per node: {avg_relations:.2f}")
                
                # Show top nodes with most relations
                print(f"\n  Top nodes with most relations:")
                for result in results[:10]:  # Top 10
                    if result['incoming_relations'] > 0:
                        print(f"    {result['name']}: {result['incoming_relations']} relations")
                        
                        # Show relation breakdown
                        relation_counts = defaultdict(int)
                        source_type_counts = defaultdict(int)
                        
                        for detail in result['relation_details']:
                            if detail['relation_type']:  # Skip None relations
                                relation_counts[detail['relation_type']] += 1
                                if detail['source_type']:
                                    source_type_counts[detail['source_type']] += 1
                        
                        if relation_counts:
                            print(f"      Relations: {dict(relation_counts)}")
                            print(f"      Sources: {dict(source_type_counts)}")
                
                # Show nodes without relations
                nodes_without = [r for r in results if r['incoming_relations'] == 0]
                if nodes_without:
                    print(f"\n  Nodes without relations ({len(nodes_without)}):")
                    for result in nodes_without[:20]:  # Show first 20
                        print(f"    {result['name']}")
                    if len(nodes_without) > 20:
                        print(f"    ... and {len(nodes_without) - 20} more")
                
                # Add to overall analysis
                analysis_results.append({
                    'node_type': node_type,
                    'total_nodes': total_nodes,
                    'nodes_with_relations': nodes_with_relations,
                    'nodes_without_relations': nodes_without_relations,
                    'max_relations': max_relations,
                    'avg_relations': avg_relations,
                    'coverage_percent': (nodes_with_relations / total_nodes * 100) if total_nodes > 0 else 0
                })
            
            # Create summary report
            print(f"\n{'='*80}")
            print("OVERALL SUMMARY")
            print('='*80)
            
            df = pd.DataFrame(analysis_results)
            print(df.to_string(index=False))
            
            # Save detailed results
            results_dir = Path('results')
            results_dir.mkdir(exist_ok=True)
            
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            csv_file = results_dir / f'node_relations_analysis_{timestamp}.csv'
            df.to_csv(csv_file, index=False)
            
            # Also save detailed breakdown for each node type
            detailed_file = results_dir / f'detailed_node_analysis_{timestamp}.txt'
            with open(detailed_file, 'w', encoding='utf-8') as f:
                f.write("DETAILED NODE RELATIONS ANALYSIS\n")
                f.write("="*80 + "\n\n")
                
                for node_type in sorted(all_node_types):
                    f.write(f"{node_type} NODES:\n")
                    f.write("-" * 40 + "\n")
                    
                    query = f"""
                    MATCH (n:{node_type})
                    OPTIONAL MATCH (source)-[r]->(n)
                    WITH n, count(r) as incoming_relations, collect({{
                        source_type: labels(source)[0],
                        relation_type: type(r),
                        source_name: source.name
                    }}) as relation_details
                    RETURN n.name as name,
                           n.id as id,
                           incoming_relations,
                           relation_details
                    ORDER BY incoming_relations DESC, n.name
                    """
                    
                    results = session.run(query).data()
                    
                    for result in results:
                        f.write(f"{result['name']}: {result['incoming_relations']} relations\n")
                        if result['incoming_relations'] > 0:
                            for detail in result['relation_details']:
                                if detail['relation_type']:
                                    f.write(f"  <- {detail['source_name']} ({detail['source_type']}) via {detail['relation_type']}\n")
                        f.write("\n")
                    
                    f.write("\n" + "="*80 + "\n\n")
            
            logger.info(f"\nResults saved to:")
            logger.info(f"  Summary: {csv_file}")
            logger.info(f"  Detailed: {detailed_file}")
            
    finally:
        driver.close()

if __name__ == "__main__":
    analyze_node_relations() 