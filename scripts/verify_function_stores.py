#!/usr/bin/env python3
"""
Verification script to check which functions are registered in Qdrant and MongoDB.
Displays a comparison of functions in both stores and identifies any discrepancies.
"""
import asyncio
from pymongo import MongoClient
from qdrant_client import QdrantClient
from tabulate import tabulate
from typing import List, Dict, Set
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ccp.core.settings import settings


def check_mongodb_functions() -> List[Dict]:
    """Check functions registered in MongoDB."""
    print("🔍 Checking MongoDB...")
    
    try:
        client = MongoClient(settings.mongo_uri)
        db = client["ccp_db"]
        collection = db["function_registry"]
        
        functions = list(collection.find({}, {"_id": 0, "name": 1, "module": 1, "description": 1}))
        
        print(f"✅ MongoDB connected. Found {len(functions)} functions.")
        return functions
    
    except Exception as e:
        print(f"❌ MongoDB error: {e}")
        return []


def check_qdrant_functions() -> List[Dict]:
    """Check functions registered in Qdrant."""
    print("\n🔍 Checking Qdrant...")
    
    try:
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if "function_registry" not in collection_names:
            print("⚠️  Qdrant collection 'function_registry' does not exist.")
            return []
        
        # Get collection info
        collection_info = client.get_collection("function_registry")
        point_count = collection_info.points_count
        
        print(f"✅ Qdrant connected. Collection has {point_count} points.")
        
        # Retrieve all points
        if point_count == 0:
            return []
        
        # Scroll through all points
        points, _ = client.scroll(
            collection_name="function_registry",
            limit=100,
            with_payload=True,
            with_vectors=False
        )
        
        functions = []
        for point in points:
            payload = point.payload
            functions.append({
                "name": payload.get("name"),
                "module": payload.get("module"),
                "description": payload.get("description", "")[:100]  # Truncate description
            })
        
        return functions
    
    except Exception as e:
        print(f"❌ Qdrant error: {e}")
        return []


def compare_stores(mongo_funcs: List[Dict], qdrant_funcs: List[Dict]):
    """Compare functions in both stores and display results."""
    print("\n" + "="*80)
    print("📊 FUNCTION REGISTRY COMPARISON")
    print("="*80)
    
    # Extract function names
    mongo_names = {f["name"] for f in mongo_funcs}
    qdrant_names = {f["name"] for f in qdrant_funcs}
    
    # Find differences
    only_in_mongo = mongo_names - qdrant_names
    only_in_qdrant = qdrant_names - mongo_names
    in_both = mongo_names & qdrant_names
    
    # Summary statistics
    print(f"\n📈 Summary:")
    print(f"  • Total in MongoDB: {len(mongo_names)}")
    print(f"  • Total in Qdrant: {len(qdrant_names)}")
    print(f"  • In both stores: {len(in_both)}")
    print(f"  • Only in MongoDB: {len(only_in_mongo)}")
    print(f"  • Only in Qdrant: {len(only_in_qdrant)}")
    
    # Display functions in both stores
    if in_both:
        print(f"\n✅ Functions in BOTH stores ({len(in_both)}):")
        table_data = []
        for name in sorted(in_both):
            mongo_func = next(f for f in mongo_funcs if f["name"] == name)
            table_data.append([
                name,
                mongo_func["module"],
                mongo_func["description"][:60] + "..." if len(mongo_func["description"]) > 60 else mongo_func["description"]
            ])
        
        print(tabulate(table_data, headers=["Function", "Module", "Description"], tablefmt="grid"))
    
    # Display discrepancies
    if only_in_mongo:
        print(f"\n⚠️  Functions ONLY in MongoDB ({len(only_in_mongo)}):")
        for name in sorted(only_in_mongo):
            mongo_func = next(f for f in mongo_funcs if f["name"] == name)
            print(f"  • {name} ({mongo_func['module']})")
    
    if only_in_qdrant:
        print(f"\n⚠️  Functions ONLY in Qdrant ({len(only_in_qdrant)}):")
        for name in sorted(only_in_qdrant):
            qdrant_func = next(f for f in qdrant_funcs if f["name"] == name)
            print(f"  • {name} ({qdrant_func['module']})")
    
    # Health check
    print("\n" + "="*80)
    if len(only_in_mongo) == 0 and len(only_in_qdrant) == 0 and len(in_both) > 0:
        print("✅ HEALTH CHECK: PASSED - Stores are synchronized!")
    elif len(in_both) == 0:
        print("❌ HEALTH CHECK: FAILED - No functions found in stores!")
    else:
        print("⚠️  HEALTH CHECK: WARNING - Stores are not fully synchronized!")
    print("="*80)


def main():
    """Main verification function."""
    print("\n" + "="*80)
    print("🔧 CCP Function Registry Verification")
    print("="*80)
    print(f"\nConnecting to:")
    print(f"  • MongoDB: {settings.mongo_uri}")
    print(f"  • Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
    print()
    
    # Check both stores
    mongo_functions = check_mongodb_functions()
    qdrant_functions = check_qdrant_functions()
    
    # Compare and display results
    compare_stores(mongo_functions, qdrant_functions)
    
    print("\n✨ Verification complete!\n")


if __name__ == "__main__":
    main()
