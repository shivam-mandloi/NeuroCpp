
with open('iris.txt', 'r') as f:
    data = f.read()

data = data.replace(",", " ")
with open('iris.txt', 'w') as f:
    f.write(data)

print("[*] Done")