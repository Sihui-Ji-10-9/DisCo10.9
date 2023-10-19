import ast

string = 'tensor([[[[-1., -1., -1.,  ..., -1., -1., -1.]]]])'

# 去掉两端括号以外的内容
result = string[string.find("[[[["):string.find("]]]]")+4]
print(type(result))
print(result)
array = ast.literal_eval(result)
print(type(array))
print(array)
