Examples
--------
>>> A              >>> B
    lkey value         rkey value
0   foo  1         0   foo  5
1   bar  2         1   bar  6
2   baz  3         2   qux  7
3   foo  4         3   bar  8
>>> A.merge(B, left_on='lkey', right_on='rkey', how='outer')
   lkey  value_x  rkey  value_y
0  foo   1        foo   5
1  foo   4        foo   5
2  bar   2        bar   6
3  bar   2        bar   8
4  baz   3        NaN   NaN
5  NaN   NaN      qux   7
Returns
-------
merged : DataFrame
    The output type will the be same as 'left', if it is a subclass
    of DataFrame.

    
    
DataFrame.merge(right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, 
                sort=False, suffixes=('_x', '_y'), copy=True, indicator=False)

Parameters:
right : DataFrame
how : {‘left’, ‘right’, ‘outer’, ‘inner’}, default ‘inner’
left: use only keys from left frame, similar to a SQL left outer join; preserve key order
right: use only keys from right frame, similar to a SQL right outer join; preserve key order
outer: use union of keys from both frames, similar to a SQL full outer join; sort keys lexicographically
inner: use intersection of keys from both frames, similar to a SQL inner join; preserve the order of the left keys
on : label or list
Field names to join on. Must be found in both DataFrames. If on is None and not merging on indexes, then it merges on the intersection of the columns by default.
left_on : label or list, or array-like
Field names to join on in left DataFrame. Can be a vector or list of vectors of the length of the DataFrame to use a particular vector as the join key instead of columns
right_on : label or list, or array-like
Field names to join on in right DataFrame or vector/list of vectors per left_on docs
left_index : boolean, default False
Use the index from the left DataFrame as the join key(s). If it is a MultiIndex, the number of keys in the other DataFrame (either the index or a number of columns) must match the number of levels
right_index : boolean, default False
Use the index from the right DataFrame as the join key. Same caveats as left_index
sort : boolean, default False
Sort the join keys lexicographically in the result DataFrame. If False, the order of the join keys depends on the join type (how keyword)
suffixes : 2-length sequence (tuple, list, ...)
Suffix to apply to overlapping column names in the left and right side, respectively
copy : boolean, default True
If False, do not copy data unnecessarily
indicator : boolean or string, default False
If True, adds a column to output DataFrame called “_merge” with information on the source of each row. If string, column with information on source of each row will be added to output DataFrame, and column will be named value of string. 
Information column is Categorical-type and takes on a value of “left_only” for observations whose merge key only appears in ‘left’ DataFrame, “right_only” for observations whose merge key only appears in ‘right’ DataFrame, and “both” 
if the observation’s merge key is found in both.


Returns:
   merged : DataFrame
The output type will the be same as ‘left’, if it is a subclass of DataFrame. 
