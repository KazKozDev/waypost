import re
with open('waypost/ui.py', 'r') as f:
    content = f.read()

# Extract the script block
m = re.search(r'<script>(.*?)</script>', content, re.DOTALL)
if m:
    js_code = m.group(1)
    # un-escape the python f-string curly braces
    js_code = js_code.replace('{{', '{').replace('}}', '}')
    with open('test.js', 'w') as out:
        out.write(js_code)
    print("Extracted to test.js")
