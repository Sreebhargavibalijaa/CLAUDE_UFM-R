import base64

with open("/Users/sreebhargavibalija/Desktop/ufmr_claude_mcp_tool/chest.jpg", "rb") as f:
    encoded = base64.b64encode(f.read()).decode('utf-8')

# Save to clipboard or file
print(encoded)
