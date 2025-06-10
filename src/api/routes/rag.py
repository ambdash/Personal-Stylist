from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List
from ..services.enhanced_rag_service import EnhancedRagService
from ..db.neo4j.config import Neo4jConnection
from pydantic import BaseModel

router = APIRouter(prefix="/rag", tags=["rag"])

class RagRequest(BaseModel):
    prompt: str

class NodeInfo(BaseModel):
    id: str
    name: str
    type: str
    score: float = 0.0

class ConceptInfo(BaseModel):
    id: str
    name: str
    relations: List[Dict[str, Any]]
    intersection_score: int = 0
    formatted_text: str

class EnhancedRagResponse(BaseModel):
    status: str
    original_prompt: str
    enhanced_prompt: str
    enhanced: bool
    strategy_used: str
    processing_time: float
    
    # Node analysis
    key_nodes: Dict[str, List[NodeInfo]]
    total_keynodes: int
    
    # Item detection
    item_type: str = ""
    item_subtype: str = ""
    is_styling: bool = False
    styling_item: str = ""
    
    # Concepts found
    concepts: List[ConceptInfo]
    concepts_count: int
    
    # Strategy details
    minimum_intersection_threshold: int = 0
    concepts_after_filtering: int = 0

class RagResponse(BaseModel):
    status: str
    extracted_nodes: List[Dict[str, Any]]
    knowledge_chunks: List[List[Dict[str, Any]]]
    prompt_additions: List[str]

class DebugRagResponse(BaseModel):
    status: str
    original_prompt: str
    enhanced_prompt: str
    enhanced: bool
    strategy_used: str
    processing_time: float
    total_time: float
    
    # Detailed analysis
    item_analysis: Dict[str, Any]
    key_nodes_analysis: Dict[str, List[Dict[str, Any]]]
    concepts_analysis: List[Dict[str, Any]]
    
    # Raw data
    raw_result: Dict[str, Any]

def get_enhanced_rag_service():
    """Get Enhanced RAG service for dependency injection"""
    return EnhancedRagService()

def get_neo4j_connection():
    """Get Neo4j connection for dependency injection"""
    return Neo4jConnection()

@router.post("/enhanced_inference", response_model=EnhancedRagResponse)
async def enhanced_rag_inference(
    request: RagRequest,
    rag_service: EnhancedRagService = Depends(get_enhanced_rag_service)
) -> EnhancedRagResponse:
    """
    Process a prompt through the enhanced RAG pipeline with detailed analysis.
    
    The enhanced pipeline:
    1. Detects item types and styling context
    2. Finds key nodes using NLP and fuzzy matching (including weather detection)
    3. Applies different strategies based on node count:
       - Strategy 1: Item type/styling context + key nodes
       - Strategy 2: Key node intersections only
       - Strategy 3: Single key node + random relations
    4. Filters concepts based on intersection scores
    5. Returns detailed analysis and enhanced prompt
    """
    try:
        result = await rag_service.process_rag_query(request.prompt)
        
        if result["status"] != "success":
            raise HTTPException(status_code=500, detail=result.get("error", "RAG processing failed"))
        
        # Convert key nodes to NodeInfo objects with enhanced debugging
        key_nodes_formatted = {}
        total_keynodes = 0
        weather_nodes_found = []
        
        for node_type, nodes in result.get("key_nodes", {}).items():
            key_nodes_formatted[node_type] = []
            for node in nodes:
                node_info = NodeInfo(
                    id=node["id"],
                    name=node["name"],
                    type=node.get("labels", [node_type])[0] if node.get("labels") else node_type,
                    score=node.get("score", 0.0)
                )
                key_nodes_formatted[node_type].append(node_info)
                
                # Track weather nodes for debugging
                if node_type == "Погода":
                    weather_nodes_found.append({
                        "name": node["name"],
                        "id": node["id"],
                        "score": node.get("score", 0.0)
                    })
            
            total_keynodes += len(nodes)
        
        # Convert concepts to ConceptInfo objects
        concepts_formatted = []
        for concept in result.get("concepts", []):
            concepts_formatted.append(ConceptInfo(
                id=concept["id"],
                name=concept["name"],
                relations=concept.get("relations", []),
                intersection_score=concept.get("intersection_score", 0),
                formatted_text=rag_service.format_concept_relations(concept)
            ))
        
        # Determine strategy used with enhanced logic
        strategy_used = "unknown"
        if result.get("item_type") or result.get("is_styling"):
            strategy_used = "item_type_or_styling_context"
        elif total_keynodes == 1:
            strategy_used = "single_keynode_random_relations"
        elif total_keynodes > 1:
            strategy_used = "keynode_intersections"
        else:
            strategy_used = "no_keynodes_found"
        
        # Calculate minimum intersection threshold
        min_threshold = rag_service._get_minimum_intersection_threshold(total_keynodes)
        
        # Enhanced response with weather debugging
        response = EnhancedRagResponse(
            status="success",
            original_prompt=result["original_prompt"],
            enhanced_prompt=result["enhanced_prompt"],
            enhanced=result["enhanced"],
            strategy_used=strategy_used,
            processing_time=result["processing_time"],
            
            # Node analysis
            key_nodes=key_nodes_formatted,
            total_keynodes=total_keynodes,
            
            # Item detection
            item_type=result.get("item_type", ""),
            item_subtype=result.get("item_subtype", ""),
            is_styling=result.get("is_styling", False),
            styling_item=result.get("styling_item", ""),
            
            # Concepts
            concepts=concepts_formatted,
            concepts_count=len(concepts_formatted),
            
            # Strategy details
            minimum_intersection_threshold=min_threshold,
            concepts_after_filtering=len([c for c in result.get("concepts", []) if c.get("intersection_score", 0) >= min_threshold])
        )
        
        # Add weather debugging information to the response if weather nodes were found
        if weather_nodes_found:
            # Add weather debug info as a custom field (this will be included in the response)
            response.weather_debug = {
                "weather_nodes_found": weather_nodes_found,
                "weather_detection_successful": True,
                "weather_node_count": len(weather_nodes_found)
            }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/inference", response_model=RagResponse)
async def rag_inference(
    request: RagRequest,
    rag_service: EnhancedRagService = Depends(get_enhanced_rag_service)
) -> Dict[str, Any]:
    """
    Process a prompt through the RAG pipeline (legacy endpoint for backward compatibility).
    """
    try:
        result = await rag_service.process_rag_query(request.prompt)
        
        if result["status"] != "success":
            raise HTTPException(status_code=500, detail=result.get("error", "RAG processing failed"))
        
        # Convert to legacy format
        extracted_nodes = []
        for node_type, nodes in result.get("key_nodes", {}).items():
            for node in nodes:
                extracted_nodes.append({
                    "id": node["id"],
                    "name": node["name"],
                    "type": node_type
                })
        
        # Convert concepts to knowledge chunks format
        knowledge_chunks = []
        for concept in result.get("concepts", []):
            chunk = {
                "concept": concept,
                "relations": concept.get("relations", [])
            }
            knowledge_chunks.append([chunk])
        
        # Generate prompt additions
        prompt_additions = []
        for concept in result.get("concepts", [])[:5]:  # Top 5
            formatted = rag_service.format_concept_relations(concept)
            prompt_additions.append(formatted)
        
        return {
            "status": "success",
            "extracted_nodes": extracted_nodes,
            "knowledge_chunks": knowledge_chunks,
            "prompt_additions": prompt_additions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/strategies")
async def get_rag_strategies():
    """
    Get information about RAG processing strategies.
    """
    return {
        "strategies": {
            "item_type_or_styling_context": {
                "description": "Used when specific clothing items, shoes, accessories are mentioned or styling verbs are detected",
                "triggers": ["платье", "обувь", "аксессуары", "стилизовать", "носить", "сочетать"],
                "logic": "Finds concepts related to the specific item type and intersects with key nodes"
            },
            "keynode_intersections": {
                "description": "Used when multiple key nodes are found but no specific item type",
                "triggers": ["multiple key nodes (Случай, Эстетика, Сезон, Тренд, Погода)"],
                "logic": "Finds concepts that intersect with multiple key nodes, applies minimum intersection threshold"
            },
            "single_keynode_random_relations": {
                "description": "Used when only one key node is found",
                "triggers": ["single key node found"],
                "logic": "Finds 5 random concepts related to the single key node"
            },
            "no_keynodes_found": {
                "description": "Used when no key nodes are detected in the prompt",
                "triggers": ["no recognizable fashion/style terms"],
                "logic": "Returns original prompt without enhancement"
            }
        },
        "key_node_types": ["Случай", "Эстетика", "Сезон", "Тренд", "Погода"],
        "item_types": {
            "обувь": ["туфли", "кроссовки", "ботинки", "сапоги", "лоферы", "балетки"],
            "одежда": ["платье", "юбка", "брюки", "джинсы", "куртка", "пальто", "блуза"],
            "аксессуары": ["сумка", "украшения", "серьги", "браслет", "шарф", "ремень"]
        }
    }

@router.post("/debug", response_model=DebugRagResponse)
async def debug_rag_processing(
    request: RagRequest,
    rag_service: EnhancedRagService = Depends(get_enhanced_rag_service)
) -> DebugRagResponse:
    """
    Debug RAG processing with detailed analysis.
    
    This endpoint provides comprehensive debugging information about how the RAG service
    processes a given prompt, including:
    - Item type and styling context detection
    - Key nodes found and their scores
    - Strategy selection logic
    - Concept matching and intersection scores
    - Detailed timing information
    
    Useful for understanding and debugging the RAG pipeline behavior.
    """
    import time
    start_time = time.time()
    
    try:
        result = await rag_service.enhance_prompt_async(request.prompt)
        total_time = time.time() - start_time
        
        if result["status"] != "success":
            raise HTTPException(status_code=500, detail=result.get("error", "RAG processing failed"))
        
        # Analyze item detection
        item_analysis = {
            "item_type": result.get("item_type", ""),
            "item_subtype": result.get("item_subtype", ""),
            "is_styling": result.get("is_styling", False),
            "styling_item": result.get("styling_item", ""),
            "detection_logic": {
                "has_item_type": bool(result.get("item_type")),
                "has_subtype": bool(result.get("item_subtype")) and result.get("item_subtype") != result.get("item_type"),
                "is_styling_context": result.get("is_styling", False)
            }
        }
        
        # Analyze key nodes
        key_nodes_analysis = {}
        key_nodes_found = result.get("key_nodes_found", [])
        
        # Group by type with detailed info
        for node in key_nodes_found:
            node_type = node.get("type", "Unknown")
            if node_type not in key_nodes_analysis:
                key_nodes_analysis[node_type] = []
            
            key_nodes_analysis[node_type].append({
                "id": node.get("id", ""),
                "name": node.get("name", ""),
                "score": node.get("score", 0.0),
                "labels": node.get("labels", [])
            })
        
        # Analyze concepts
        concepts_analysis = []
        for concept in result.get("concepts_used", []):
            concept_analysis = {
                "id": concept.get("id", ""),
                "name": concept.get("name", ""),
                "display_name": concept.get("name", "").split(':', 1)[1] if ':' in concept.get("name", "") else concept.get("name", ""),
                "intersection_score": concept.get("intersection_score", 0),
                "relations_count": len(concept.get("relations", [])),
                "relations": concept.get("relations", []),
                "formatted_text": concept.get("formatted_text", ""),
                "score_category": (
                    "high" if concept.get("intersection_score", 0) >= 2 
                    else "medium" if concept.get("intersection_score", 0) >= 1 
                    else "low"
                )
            }
            concepts_analysis.append(concept_analysis)
        
        return DebugRagResponse(
            status="success",
            original_prompt=result["original_prompt"],
            enhanced_prompt=result["enhanced_prompt"],
            enhanced=result["enhanced"],
            strategy_used=result["strategy_used"],
            processing_time=result["processing_time"],
            total_time=total_time,
            
            item_analysis=item_analysis,
            key_nodes_analysis=key_nodes_analysis,
            concepts_analysis=concepts_analysis,
            
            raw_result=result
        )
        
    except Exception as e:
        total_time = time.time() - start_time
        raise HTTPException(status_code=500, detail={
            "error": str(e),
            "processing_time": total_time
        }) 