path = r'C:\Ahmed Bayoumi\University\ZC Hackathon\run_pipeline.py'
with open(path, 'rb') as f:
    content = f.read()
# smart left/right double quotes -> straight double quote
content = content.replace(b'\xe2\x80\x9c', b'"').replace(b'\xe2\x80\x9d', b'"')
# smart left/right single quotes -> straight single quote
content = content.replace(b'\xe2\x80\x98', b"'").replace(b'\xe2\x80\x99', b"'")
with open(path, 'wb') as f:
    f.write(content)
print('Fixed curly quotes in run_pipeline.py')
