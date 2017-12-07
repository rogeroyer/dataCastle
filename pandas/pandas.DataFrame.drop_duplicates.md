```python
# 删除content和tags相同的行 #
read_data = read_data.drop_duplicates(['content', 'tags'])
```
