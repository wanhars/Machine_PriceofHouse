import chardet

with open('E:/code/Pycharm/PythonProject1/data/new.csv', 'rb') as f:
    result = chardet.detect(f.read(10000))
    print(result)
