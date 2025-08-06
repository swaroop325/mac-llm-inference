#!/usr/bin/env python3
"""
Export OpenAPI specification to JSON file
"""

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

def export_openapi():
    """Export OpenAPI spec to JSON file"""
    openapi_schema = app.openapi()
    
    # Write to file
    with open('openapi.json', 'w') as f:
        json.dump(openapi_schema, f, indent=2)
    
    print("✅ OpenAPI specification exported to openapi.json")
    print(f"📄 Title: {openapi_schema['info']['title']}")
    print(f"📋 Version: {openapi_schema['info']['version']}")
    print(f"🔗 Endpoints: {len(openapi_schema['paths'])} paths")

if __name__ == "__main__":
    export_openapi()