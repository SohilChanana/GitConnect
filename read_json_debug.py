import json
import os

try:
    # PowerShell Out-File defaults to utf-16
    with open('debug_response.json', 'r', encoding='utf-16') as f:
        data = json.load(f)
except Exception:
    # Fallback to utf-8 if something else wrote it
    with open('debug_response.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

print(f"Total Relevant Entities: {len(data['relevant_entities'])}")
print("-" * 60)
print(f"{'Type':<10} | {'Name':<30} | {'Has Content'}")
print("-" * 60)

count_with_content = 0
for e in data['relevant_entities']:
    has_content = bool(e.get('content'))
    if has_content: count_with_content += 1
    # Truncate name for display
    name = (e.get('name') or "N/A")[:28]
    fpath = (e.get('file_path') or "N/A")[-30:]
    print(f"{e.get('type', 'Unknown'):<10} | {name:<30} | {fpath:<30} | {has_content}")

print("-" * 60)
print(f"Entities with content: {count_with_content}/{len(data['relevant_entities'])}")
