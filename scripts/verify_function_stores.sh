#!/bin/bash
# Wrapper script to run function store verification inside Docker container

echo "🔧 Running Function Store Verification..."
echo ""

docker compose exec backend python3 -c "
import sys
from pymongo import MongoClient
from qdrant_client import QdrantClient
from src.ccp.core.settings import settings

def check_mongodb_functions():
    print('🔍 Checking MongoDB...')
    try:
        client = MongoClient(settings.mongo_uri)
        db = client['ccp_storage']  # FIXED: Use correct database
        collection = db['function_metadata']
        functions = list(collection.find({}, {'_id': 0, 'name': 1, 'docstring': 1}))
        print(f'✅ MongoDB connected. Found {len(functions)} functions.')
        return functions
    except Exception as e:
        print(f'❌ MongoDB error: {e}')
        return []

def check_qdrant_functions():
    print('\n🔍 Checking Qdrant...')
    try:
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if 'function_registry' not in collection_names:
            print('⚠️  Qdrant collection does not exist.')
            return []
        
        # Use scroll instead of get_collection to avoid validation errors
        points, _ = client.scroll(
            collection_name='function_registry',
            limit=100,
            with_payload=True,
            with_vectors=False
        )
        
        print(f'✅ Qdrant connected. Collection has {len(points)} points.')
        
        functions = []
        for point in points:
            payload = point.payload
            functions.append({
                'name': payload.get('name'),
                'docstring': payload.get('docstring', '')[:100]
            })
        return functions
    except Exception as e:
        print(f'❌ Qdrant error: {e}')
        return []

def compare_stores(mongo_funcs, qdrant_funcs):
    print('\n' + '='*80)
    print('📊 FUNCTION REGISTRY COMPARISON')
    print('='*80)
    
    mongo_names = {f['name'] for f in mongo_funcs}
    qdrant_names = {f['name'] for f in qdrant_funcs}
    
    only_in_mongo = mongo_names - qdrant_names
    only_in_qdrant = qdrant_names - mongo_names
    in_both = mongo_names & qdrant_names
    
    print(f'\n📈 Summary:')
    print(f'  • Total in MongoDB: {len(mongo_names)}')
    print(f'  • Total in Qdrant: {len(qdrant_names)}')
    print(f'  • In both stores: {len(in_both)}')
    print(f'  • Only in MongoDB: {len(only_in_mongo)}')
    print(f'  • Only in Qdrant: {len(only_in_qdrant)}')
    
    if in_both:
        print(f'\n✅ Functions in BOTH stores ({len(in_both)}):')
        print('-' * 80)
        for name in sorted(in_both):
            mongo_func = next(f for f in mongo_funcs if f['name'] == name)
            desc = mongo_func['docstring'][:60] + '...' if len(mongo_func['docstring']) > 60 else mongo_func['docstring']
            print(f'  📦 {name}')
            if desc:
                print(f'     {desc}')
    
    if only_in_mongo:
        print(f'\n⚠️  Functions ONLY in MongoDB ({len(only_in_mongo)}):')
        for name in sorted(only_in_mongo):
            print(f'  • {name}')
    
    if only_in_qdrant:
        print(f'\n⚠️  Functions ONLY in Qdrant ({len(only_in_qdrant)}):')
        for name in sorted(only_in_qdrant):
            print(f'  • {name}')
    
    print('\n' + '='*80)
    if len(only_in_mongo) == 0 and len(only_in_qdrant) == 0 and len(in_both) > 0:
        print('✅ HEALTH CHECK: PASSED - Stores are synchronized!')
    elif len(in_both) == 0:
        print('❌ HEALTH CHECK: FAILED - No functions found in stores!')
    else:
        print('⚠️  HEALTH CHECK: WARNING - Stores are not fully synchronized!')
    print('='*80)

print('\n' + '='*80)
print('🔧 CCP Function Registry Verification')
print('='*80)
print(f'\nConnecting to:')
print(f'  • MongoDB: {settings.mongo_uri} (ccp_storage.function_metadata)')
print(f'  • Qdrant: {settings.qdrant_host}:{settings.qdrant_port} (function_registry)')
print()

mongo_functions = check_mongodb_functions()
qdrant_functions = check_qdrant_functions()
compare_stores(mongo_functions, qdrant_functions)

print('\n✨ Verification complete!\n')
"
